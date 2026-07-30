"""
ActiveInference.py
=========================
Minimum-viable PoC: Active Inference agent for network-level cyber-defence.

------------------------------------------------------
1. OU (mean-reverting) environment
       Lo stato benigno evolve attorno a un valore di riferimento fisso seguendo un processo di Ornstein-Uhlenbeck

2. No episode-terminating action
       Tutte e tre le azioni modificano lo stato; ogni episodio viene eseguito per un numero fisso di passi.
            Azione 0 – monitoraggio: riduce il rumore delle osservazioni (con una lieve modifica dello stato).
            Azione 1 – nessuna operazione (no-op): non produce alcun effetto.
            Azione 2 – mitigazione:
                - se è presente un attacco → riporta rapidamente lo stato verso il valore di riferimento (baseline);
                - se lo stato è normale → comporta un piccolo costo dovuto a un falso allarme.

3. Normalised EFE components
       Il termine pragmatico e quello epistemico sono entrambi normalizzati dividendo ciascuno per GOAL_STD
       prima della loro somma, così da renderli confrontabili sulla stessa scala.

Key convergence test
--------------------
Al termine dell'addestramento, la probabilità di scegliere l'azione di mitigazione in presenza di un attacco (P(mitigate∣attack))
dovrebbe risultare significativamente superiore alla probabilità di scegliere la stessa azione in condizioni normali (P(mitigate∣benign)).
Tale confronto viene mostrato al termine dell'addestramento.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import random
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────
STATE_DIM  = 3
ACTION_DIM = 3
HIDDEN_DIM = 32

# Homeostatic zone centre and tolerance (latency ms, packet_loss %, packet_rate)
GOAL_MEAN = np.array([35.0, 2.5, 75.0])
GOAL_STD  = np.array([ 5.0, 1.0, 10.0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
class NetworkEnv:
    """
    Processo di Ornstein-Uhlenbeck tridimensionale attorno a GOAL_MEAN.
    Modalità nascosta: stato normale (False) | sotto attacco (True).

    L'attacco introduce una deriva persistente verso valori più elevati,
    mentre l'azione 2 (mitigazione) la contrasta, riportando lo stato verso il valore di riferimento.
    """
    OU_THETA  = 0.25
    OU_SIGMA  = np.array([1.0, 0.25, 2.0])     # benign noise (1-sigma ≈ 0.3 GOAL_STD)
    ATK_RATE  = np.array([5.0, 1.5, 10.0])     # attack drift → equilibrium ≈ 4 GOAL_STD

    def __init__(self, attack_prob=0.04, max_steps=100):
        self.attack_prob = attack_prob
        self.max_steps   = max_steps
        self.reset()

    def reset(self):
        # Start near baseline with small random perturbation
        self.state        = GOAL_MEAN + np.random.normal(0, GOAL_STD * 0.3)
        self.noise_level  = 1.0
        self.attack       = False
        self.current_step = 0
        return self._obs(), self.state.copy()

    def step(self, action):
        self.current_step += 1

        # --- Action effects ---
        if action == 0:   # monitor: reduce noise, tiny state bump
            self.noise_level = max(0.2, self.noise_level * 0.8)
            self.state += np.random.uniform(0.1, 0.5, 3)

        elif action == 2:  # mitigate
            if self.attack:
                # Strong pull back toward baseline; 50 % chance attack resolves
                self.state = self.state + 0.6 * (GOAL_MEAN - self.state)
                if np.random.rand() < 0.5:
                    self.attack = False
            else:
                # False-alarm cost
                self.state += np.random.uniform(0.5, 1.5, 3)

        # action 1: no-op — nothing

        # --- OU mean-reverting benign dynamics ---
        dW = np.random.normal(0, 1, 3)
        self.state += self.OU_THETA * (GOAL_MEAN - self.state) + self.OU_SIGMA * dW

        # --- Attack dynamics ---
        if not self.attack and np.random.rand() < self.attack_prob:
            self.attack = True
        if self.attack:
            self.state += self.ATK_RATE + np.random.uniform(0, 0.5, 3)

        # Physical bounds
        self.state = np.clip(self.state, [5, 0, 0], [200, 100, 300])

        done = (self.current_step >= self.max_steps)
        return self._obs(), self.state.copy(), done, self.attack

    def _obs(self):
        return self.state + np.random.normal(0, self.noise_level, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Neural networks
# ─────────────────────────────────────────────────────────────────────────────
class PosteriorNet(nn.Module):
    """Maps (obs, prev_action_one_hot) → inferred hidden state."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),             nn.ReLU(),
            nn.Linear(HIDDEN_DIM, STATE_DIM),
        )
    def forward(self, obs, a_oh):
        return self.net(torch.cat([obs, a_oh], dim=-1))


class PolicyNet(nn.Module):
    """Maps inferred state → action logits."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM),
        )
    def forward(self, s):
        return self.net(s)


class CriticNet(nn.Module):
    """Maps inferred state → Expected Free Energy per action."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM),
        )
    def forward(self, s):
        return self.net(s)


# ─────────────────────────────────────────────────────────────────────────────
# Active Inference Agent
# ─────────────────────────────────────────────────────────────────────────────
class ActiveInferenceAgent:
    def __init__(self, lr=3e-4, gamma=0.6, zeta=8.0):
        self.gamma  = gamma
        self.zeta   = zeta
        self.memory = deque(maxlen=20000)

        self.posterior     = PosteriorNet().to(device)
        self.policy        = PolicyNet().to(device)
        self.critic        = CriticNet().to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.eval()

        self.opt_post = optim.Adam(self.posterior.parameters(), lr=lr)
        self.opt_pol  = optim.Adam(self.policy.parameters(),    lr=lr)
        self.opt_crit = optim.Adam(self.critic.parameters(),    lr=lr)

        self.goal_mean = torch.FloatTensor(GOAL_MEAN).to(device)
        self.goal_std  = torch.FloatTensor(GOAL_STD).to(device)

    def _norm(self, s_tensor):
        """Normalise raw state to zero-centred units of GOAL_STD."""
        return (s_tensor - self.goal_mean) / self.goal_std

    def infer(self, obs, a):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        a_oh  = F.one_hot(torch.tensor([a]), ACTION_DIM).float().to(device)
        with torch.no_grad():
            return self.posterior(obs_t, a_oh).squeeze(0).cpu().numpy()

    def act(self, s_inf):
        # Policy and critic operate on normalised deviation, not raw values
        s_norm = self._norm(torch.FloatTensor(s_inf).unsqueeze(0).to(device))
        with torch.no_grad():
            probs = F.softmax(self.policy(s_norm), dim=-1)
        return torch.multinomial(probs, 1).item()

    def remember(self, s_t, a_prev, a_t, s_t1, o_t, o_t1):
        self.memory.append((s_t.copy(), int(a_prev), int(a_t),
                            s_t1.copy(), o_t.copy(), o_t1.copy()))

    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None, None, None

        batch  = random.sample(self.memory, batch_size)
        s_t, a_prev, a_t, s_t1, o_t, o_t1 = zip(*batch)

        s_t    = torch.FloatTensor(np.array(s_t)).to(device)
        a_prev = torch.LongTensor(np.array(a_prev)).to(device)
        a_t    = torch.LongTensor(np.array(a_t)).to(device)
        s_t1   = torch.FloatTensor(np.array(s_t1)).to(device)
        o_t    = torch.FloatTensor(np.array(o_t)).to(device)
        o_t1   = torch.FloatTensor(np.array(o_t1)).to(device)

        a_prev_oh = F.one_hot(a_prev, ACTION_DIM).float()
        a_t_oh    = F.one_hot(a_t,    ACTION_DIM).float()

        # ── Posterior: map (obs, a_prev) → inferred state ─────────────────
        mu_t      = self.posterior(o_t, a_prev_oh)
        post_loss = 0.5 * F.mse_loss(mu_t, s_t)
        self.opt_post.zero_grad()
        post_loss.backward()
        self.opt_post.step()

        # Normalised inferred states (input to policy and critic)
        mu_t_norm = self._norm(mu_t.detach())

        # ── Policy: match action distribution to softmax(-zeta * EFE) ─────
        with torch.no_grad():
            efe          = self.critic(mu_t_norm)
            target_probs = F.softmax(-self.zeta * efe, dim=-1)

        pol_logits = self.policy(mu_t_norm)
        pol_loss   = F.kl_div(F.log_softmax(pol_logits, dim=-1),
                              target_probs, reduction='batchmean')
        self.opt_pol.zero_grad()
        pol_loss.backward()
        self.opt_pol.step()

        # ── Critic: Bellman update on EFE ─────────────────────────────────
        with torch.no_grad():
            mu_t1      = self.posterior(o_t1, a_t_oh)
            mu_t1_norm = self._norm(mu_t1)

            norm_obs_dev = (o_t1 - self.goal_mean) / self.goal_std
            pragmatic    = 0.5 * (norm_obs_dev ** 2).sum(dim=-1)
            epistemic = 0.5 * (((o_t1 - mu_t1) / self.goal_std) ** 2).sum(dim=-1)

            one_step_efe  = pragmatic + epistemic
            next_efe_vals = self.target_critic(mu_t1_norm)
            min_next, _   = next_efe_vals.min(dim=-1)
            target_efe    = one_step_efe + self.gamma * min_next

        pred_efe  = self.critic(mu_t_norm.detach())
        pred_efe  = pred_efe.gather(1, a_t.unsqueeze(1)).squeeze()
        crit_loss = F.smooth_l1_loss(pred_efe, target_efe)
        self.opt_crit.zero_grad()
        crit_loss.backward()
        self.opt_crit.step()

        return post_loss.item(), pol_loss.item(), crit_loss.item()

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
NUM_EPISODES     = 400
MAX_STEPS_PER_EP = 100
BATCH_SIZE       = 64
TARGET_UPDATE_EP = 5    # update target critic every N episodes

env   = NetworkEnv(attack_prob=0.04, max_steps=MAX_STEPS_PER_EP)
agent = ActiveInferenceAgent(lr=3e-4, gamma=0.6, zeta=8.0)

ep_post_losses, ep_pol_losses, ep_crit_losses = [], [], []
ep_mean_dev, ep_attack_rate = [], []
# Conditional mitigation: P(a=2 | attack) and P(a=2 | benign) per episode
ep_mit_atk, ep_mit_ben = [], []

print(f"Training for {NUM_EPISODES} episodes × {MAX_STEPS_PER_EP} steps each…\n")

for ep in range(1, NUM_EPISODES + 1):
    obs, true_s = env.reset()
    a_prev = 1                              # start with no-op

    post_ls, pol_ls, crit_ls = [], [], []
    deviations  = []
    atk_steps   = []
    mit_when_atk, mit_when_ben = [], []

    for _ in range(MAX_STEPS_PER_EP):
        s_inf = agent.infer(obs, a_prev)
        a     = agent.act(s_inf)

        obs2, true_s2, done, attack = env.step(a)
        agent.remember(true_s, a_prev, a, true_s2, obs, obs2)

        pl, poll, cl = agent.learn(BATCH_SIZE)
        if pl is not None:
            post_ls.append(pl); pol_ls.append(poll); crit_ls.append(cl)

        dev = float(np.mean(np.abs((true_s2 - GOAL_MEAN) / GOAL_STD)))
        deviations.append(dev)
        atk_steps.append(int(attack))

        if attack:
            mit_when_atk.append(int(a == 2))
        else:
            mit_when_ben.append(int(a == 2))

        obs, true_s, a_prev = obs2, true_s2, a
        if done:
            break

    if ep % TARGET_UPDATE_EP == 0:
        agent.update_target()

    ep_post_losses.append(np.mean(post_ls)   if post_ls   else 0.0)
    ep_pol_losses .append(np.mean(pol_ls)    if pol_ls    else 0.0)
    ep_crit_losses.append(np.mean(crit_ls)   if crit_ls   else 0.0)
    ep_mean_dev   .append(np.mean(deviations))
    ep_attack_rate.append(np.mean(atk_steps))
    ep_mit_atk    .append(np.mean(mit_when_atk) if mit_when_atk else 0.0)
    ep_mit_ben    .append(np.mean(mit_when_ben) if mit_when_ben else 0.0)

    if ep % 50 == 0 or ep == 1:
        print(f"Ep {ep:4d} | dev={ep_mean_dev[-1]:.3f} | "
              f"atk={ep_attack_rate[-1]:.2f} | "
              f"mit|atk={ep_mit_atk[-1]:.2f}  mit|ben={ep_mit_ben[-1]:.2f} | "
              f"post={ep_post_losses[-1]:.4f}  pol={ep_pol_losses[-1]:.4f}  "
              f"crit={ep_crit_losses[-1]:.4f}")

print("\nTraining complete.")

# ─────────────────────────────────────────────────────────────────────────────
# Convergence summary
# ─────────────────────────────────────────────────────────────────────────────
TAIL = 50   # last N episodes for summary statistics
mit_atk_final = np.mean(ep_mit_atk[-TAIL:])
mit_ben_final = np.mean(ep_mit_ben[-TAIL:])
dev_final     = np.mean(ep_mean_dev[-TAIL:])

print("\n" + "=" * 60)
print("CONVERGENCE SUMMARY  (last 50 episodes)")
print("=" * 60)
print(f"  Mean normalised deviation from homeostatic zone : {dev_final:.3f}")
print(f"  P(mitigate | attack)                           : {mit_atk_final:.3f}")
print(f"  P(mitigate | benign)                           : {mit_ben_final:.3f}")
ratio = (mit_atk_final / mit_ben_final) if mit_ben_final > 1e-6 else float('inf')
print(f"  Selectivity ratio P(mit|atk)/P(mit|ben)        : {ratio:.2f}x")
if ratio > 2.0:
    print("  -> Agent has learned to preferentially mitigate attacks. CONVERGED.")
else:
    print("  -> Ratio < 2: agent not yet selective. More training may be needed.")

# ─────────────────────────────────────────────────────────────────────────────
# Convergence plots
# ─────────────────────────────────────────────────────────────────────────────
W = 20   # smoothing window

def smooth(x, w=W):
    return np.convolve(x, np.ones(w) / w, mode='valid')

eps = np.arange(1, NUM_EPISODES + 1)

fig, axs = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Active Inference PoC — Training Convergence", fontsize=13)

def _plot(ax, data, title, ylabel="", ref=None):
    ax.plot(eps, data, alpha=0.25, color='steelblue')
    if len(data) >= W:
        ax.plot(eps[W-1:], smooth(data), color='steelblue', linewidth=2)
    if ref is not None:
        ax.axhline(ref, linestyle='--', color='gray', alpha=0.6)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.4)

_plot(axs[0, 0], ep_post_losses,  "Posterior Loss")
_plot(axs[0, 1], ep_pol_losses,   "Policy Loss (KL)")
_plot(axs[1, 0], ep_crit_losses,  "Critic Loss (EFE Bellman)")
_plot(axs[1, 1], ep_mean_dev,     "Mean Normalised Deviation\nfrom Homeostatic Zone",
      ref=1.0)
_plot(axs[2, 0], ep_mit_atk,
      "P(mitigate | attack)\nKey convergence signal", ref=0.33)
axs[2, 0].plot(eps, ep_mit_ben, alpha=0.25, color='tomato')
if len(ep_mit_ben) >= W:
    axs[2, 0].plot(eps[W-1:], smooth(ep_mit_ben), color='tomato',
                   linewidth=2, label='benign')
axs[2, 0].legend(['attack', '', 'benign', ''], fontsize=8)

_plot(axs[2, 1], ep_attack_rate,  "Attack Fraction per Episode")

for ax in axs.flat:
    ax.set_xlabel("Episode")

plt.tight_layout()
plt.savefig("convergence.png", dpi=150)
print("\nSaved convergence.png")

# ─────────────────────────────────────────────────────────────────────────────
# Test episode — greedy policy (argmin EFE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST EPISODE  (greedy policy)")
print("=" * 60)

obs, true_s = env.reset()
traj, atk_flags, actions_taken = [true_s.copy()], [env.attack], []
a_prev = 1

for _ in range(MAX_STEPS_PER_EP):
    s_inf      = agent.infer(obs, a_prev)
    s_norm_t   = agent._norm(torch.FloatTensor(s_inf).unsqueeze(0).to(device))
    with torch.no_grad():
        efe_vals = agent.critic(s_norm_t).squeeze(0).cpu().numpy()
    a = int(np.argmin(efe_vals))            # greedy: minimum EFE action

    obs, true_s, done, attack = env.step(a)
    traj.append(true_s.copy())
    atk_flags.append(attack)
    actions_taken.append(a)
    a_prev = a
    if done:
        break

traj    = np.array(traj)
mit_atk = np.mean([a == 2 for a, f in zip(actions_taken, atk_flags[1:]) if f])
mit_ben = np.mean([a == 2 for a, f in zip(actions_taken, atk_flags[1:]) if not f])
print(f"Steps: {len(traj)-1}  | "
      f"Attack fraction: {np.mean(atk_flags):.2f}  | "
      f"P(mit|atk)={mit_atk:.2f}  P(mit|ben)={mit_ben:.2f}")

names = ["Latency (ms)", "Packet Loss (%)", "Packet Rate"]
fig2, axs2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig2.suptitle("Test Episode — Greedy Policy", fontsize=13)
steps_x = np.arange(len(traj))

for i, (ax, name) in enumerate(zip(axs2, names)):
    ax.plot(steps_x, traj[:, i], color='steelblue')
    ax.axhline(GOAL_MEAN[i], linestyle='--', color='green',
               alpha=0.7, label='baseline')
    ax.axhspan(GOAL_MEAN[i] - GOAL_STD[i], GOAL_MEAN[i] + GOAL_STD[i],
               alpha=0.08, color='green')
    for j, flag in enumerate(atk_flags):
        if flag:
            ax.axvspan(j, j + 1, alpha=0.15, color='red')
    for j, ac in enumerate(actions_taken):
        if ac == 2:
            ax.axvline(j + 1, color='blue', alpha=0.5, linewidth=0.8)
    ax.set_title(name)
    ax.set_ylabel(name)
    ax.grid(True, alpha=0.4)
    ax.legend(['state', 'baseline ± 1σ', '', 'mitigate'], fontsize=8)

axs2[-1].set_xlabel("Step")
plt.tight_layout()
plt.savefig("test_episode.png", dpi=150)
print("Saved test_episode.png")

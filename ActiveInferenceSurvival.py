import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import random
from collections import deque
import wandb

# Dimensioni dell'ambiente
STATE_DIM = 3  # packet_rate, packet_loss, latency
ACTION_DIM = 3  # 0: riduci rumore, 1: nessuna azione, 2: blocca IP
HIDDEN_DIM = 128
gamma=0.9
zeta=10
first_signals= [0,0,0]

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# ENVIRONMENT
# =============================================================================

class NetworkEnvironment:
    """
    Ambiente POMDP che simula un sistema computer-router.

    Il router invia:
    - packet_rate
    - packet_loss
    - latency

    Le osservazioni sono rumorose.

    Azioni:
    0: riduci rumore (scansione)
    1: nessuna azione
    2: blocca IP (termina episodio)
    """

    def __init__(self):
        self.state = None
        self.noise_level = None
        self.attack = False
        self.attack_started = False
        self.attack_start_step = None
        self.current_step = 0
        self.action_space_size = 3
        self.reset()

    def reset(self):
        """
        Inizializza un nuovo episodio.
        """
        global first_signals
        self.current_step = 0
        # Stato normale
        self.state = np.array([75,2.5,35])
           # np.random.uniform(50, 100),  # packet_rate
           # np.random.uniform(0, 5),  # packet_loss %
           # np.random.uniform(10, 50)  # latency
        #])
        first_signals=self.state.copy()

        # Rumore iniziale
        self.noise_level = np.random.uniform(0.5, 2.0)

        # Possibile attacco
        self.attack = False
        self.attack_started = False
        self.attack_start_step = None

        if self.attack:
            # Alterazione dovuta ad attacco
            self.state += np.array([
                np.random.uniform(10, 50),  # packet_rate
                np.random.uniform(2, 15),  # packet_loss %
                np.random.uniform(10, 50)  # latency ms
            ])

        return self._get_observation(), self.state.copy()

    def step(self, action):
        done = False
        self.current_step += 1

        # --- Azioni ---
        if action == 0:  # riduzione rumore
            self.noise_level = max(0.1, self.noise_level * 0.7) # azione

            numero = random.randint(0, 2) # conseguenza
            if numero == 0:
                self.state += np.array([ np.random.uniform(1, 2), 0, 0]) # packet_rate
            elif numero == 1:
                self.state += np.array([ 0, np.random.uniform(0.5, 1), 0]) # packet_loss
            elif numero == 2:
                self.state += np.array([ 0, 0, np.random.uniform(2, 5)]) # latency

        elif action == 1:  # nessuna azione
            pass
        elif action == 2:  # termina episodio
            done = True
            obs = self._get_observation()
            return obs, self.state.copy(), done, self.attack

        # --- Evoluzione dinamica naturale ---
        drift = np.random.normal(0, [2, 0.5, 3])
        self.state += drift

        # --- Controllo attacco ---
        if not self.attack_started and np.random.rand() < 0.05:
            self.attack_started = True
            self.attack = True
            self.attack_start_step = self.current_step

        # --- Evoluzione lenta dell'attacco ---
        if self.attack:
            self.state += np.array([
                np.random.uniform(1, 5),  # packet_rate
                np.random.uniform(0.5, 2),  # packet_loss
                np.random.uniform(2, 10)  # latency
            ])

        # Vincoli fisici
        # self.state[0] = np.clip(self.state[0], 20, 180)  # packet_rate realistico
        self.state[1] = np.clip(self.state[1], 0, 100)  # packet_loss %
        # self.state[2] = np.clip(self.state[2], 5, 120)  # latency ms

        return self._get_observation(), self.state.copy(), done, self.attack

    def _get_observation(self):
        """
        Restituisce osservazione rumorosa (VFE).
        """
        noise = np.random.normal(0, self.noise_level, size=3)
        return self.state + noise


# =============================================================================
# NEURAL NETWORKS
# =============================================================================

class PosteriorNet(nn.Module):
    """Infer hidden state from observation and previous action."""

    def __init__(self, obs_dim=STATE_DIM, action_dim=ACTION_DIM, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, state_dim)

    def forward(self, obs, action_one_hot):
        x = torch.cat([obs, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc_mean(x)


class PolicyNet(nn.Module):
    """Choose actions based on inferred state."""

    def __init__(self, state_dim=STATE_DIM, num_actions=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(state_dim, hidden_dim)
        self.fc_logits = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(state))
        return self.fc_logits(x)


class CriticNet(nn.Module):
    """Estimate Expected Free Energy for each action."""

    def __init__(self, state_dim=STATE_DIM, num_actions=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(state_dim, hidden_dim)
        self.fc_efe = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(state))
        return self.fc_efe(x)


# =============================================================================
# ACTIVE INFERENCE AGENT PER NETWORK ENVIRONMENT
# =============================================================================

class ActiveInferenceAgent:
    """Agente di Inference Attiva per NetworkEnvironment."""

    def __init__(self, state_size=STATE_DIM, action_size=ACTION_DIM, lr=1e-4, gamma=gamma, zeta=zeta):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.zeta = zeta
        self.memory = deque(maxlen=10000)

        # Reti neurali
        self.posterior_net = PosteriorNet(obs_dim=state_size, action_dim=action_size, state_dim=state_size).to(device)
        self.policy_net = PolicyNet(state_dim=state_size, num_actions=action_size).to(device)
        self.critic_net = CriticNet(state_dim=state_size, num_actions=action_size).to(device)

        # Target network per stabilità
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.target_critic_net.eval()

        # Ottimizzatori
        self.posterior_optimizer = optim.Adam(self.posterior_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)

    def infer_state(self, observation, last_action):
        """Inferisce lo stato nascosto dall'osservazione e dall'ultima azione."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        action_one_hot = F.one_hot(torch.tensor([last_action]), num_classes=self.action_size).float().to(device)

        with torch.no_grad():
            inferred_state = self.posterior_net(obs_tensor, action_one_hot)
        return inferred_state.squeeze(0).cpu().numpy()

    def choose_action(self, inferred_state):
        """Select action using the policy network."""
        state_tensor = torch.FloatTensor(inferred_state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            action_probs = F.softmax(logits, dim=-1)

        return torch.multinomial(action_probs, 1).item()

    def update_target_critic(self):
        """Aggiorna la target network copiando i pesi dalla critic net."""
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

    def remember(self, true_s_t, a_t_minus_1, a_t, true_s_t_plus_1, obs_t, obs_t_plus_1, goal_signal):
        """Salva la transizione nella memoria."""

        self.memory.append((true_s_t, a_t_minus_1, a_t, true_s_t_plus_1, obs_t, obs_t_plus_1, goal_signal))

    def replay(self, batch_size):
        """Allena le reti su un batch di esperienze."""
        if len(self.memory) < batch_size:
            return None, None, None

        minibatch = random.sample(self.memory, batch_size)
        true_s_ts, a_t_minus_1s, a_ts, true_s_t_plus_1s, obs_ts, obs_t_plus_1s, goal_signals = zip(*minibatch)

        # Conversione in tensori
        true_s_t = torch.FloatTensor(np.array(true_s_ts)).to(device)
        a_t_minus_1 = torch.LongTensor(np.array(a_t_minus_1s)).to(device)
        a_t = torch.LongTensor(np.array(a_ts)).to(device)
        true_s_t_plus_1 = torch.FloatTensor(np.array(true_s_t_plus_1s)).to(device)
        obs_t = torch.FloatTensor(np.array(obs_ts)).to(device)
        obs_t_plus_1 = torch.FloatTensor(np.array(obs_t_plus_1s)).to(device)
        goal_mean = torch.FloatTensor(first_signals).to(device)  # stato normale atteso
        goal_std = torch.FloatTensor([10, 2, 10]).to(device)

        # One-hot actions
        a_t_minus_1_oh = F.one_hot(a_t_minus_1, num_classes=self.action_size).float()
        a_t_oh = F.one_hot(a_t, num_classes=self.action_size).float()

        # ===== Posterior Net =====
        inferred_s_t = self.posterior_net(obs_t, a_t_minus_1_oh)
        posterior_loss = 0.5 * F.mse_loss(inferred_s_t, true_s_t)
        self.posterior_optimizer.zero_grad()
        posterior_loss.backward()
        self.posterior_optimizer.step()

        # ===== Policy Net =====
        with torch.no_grad():
            efe_estimates = self.critic_net(inferred_s_t.detach())
            target_action_probs = F.softmax(-self.zeta * efe_estimates, dim=-1)

        policy_logits = self.policy_net(inferred_s_t.detach())
        current_action_probs = F.softmax(policy_logits, dim=-1)
        policy_loss = F.smooth_l1_loss(current_action_probs, target_action_probs)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ===== Critic Net =====
        with torch.no_grad():
            inferred_s_t_plus_1 = self.posterior_net(obs_t_plus_1, a_t_oh)

            # Epistemic gain: negative distance between true and inferred state
            epistemic_gain = -0.5 * torch.sum((true_s_t_plus_1 - inferred_s_t_plus_1) ** 2, dim=1)

            # Pragmatic penalty: distance to goal
            pragmatic_penalty = 0.5 * torch.sum(((obs_t_plus_1 - goal_mean) / goal_std) ** 2, dim=1)


            #one_step_efe = epistemic_gain + pragmatic_penalty
            # ---- Survival incentive ----
            beta = 4

            # Se sei lontano dal goal → penalità (probabile attacco)
            distance_to_goal = torch.sum(((obs_t_plus_1 - goal_mean) / goal_std) ** 2, dim=1)

            # reward positivo se vicino al goal, negativo se lontano
            survival_term = beta * torch.exp(-distance_to_goal)

            one_step_efe = epistemic_gain + pragmatic_penalty - survival_term

            next_efe_values = self.target_critic_net(inferred_s_t_plus_1)
            min_next_efe, _ = torch.min(next_efe_values, dim=1)
            target_efe = one_step_efe + self.gamma * min_next_efe

        predicted_efe_values = self.critic_net(inferred_s_t.detach())
        predicted_efe = predicted_efe_values.gather(1, a_t.unsqueeze(1)).squeeze()
        critic_loss = F.smooth_l1_loss(predicted_efe, target_efe) # cambiata da mse_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if pos_losses is not None:
            wandb.log({
                "batch_posterior_loss": pos_losses,
                "batch_policy_loss": pol_losses,
                "batch_critic_loss": c_losses
            })

        return posterior_loss.item(), policy_loss.item(), critic_loss.item()


# =============================================================================
# TRAINING
# =============================================================================

# Crea agente e ambiente
agent = ActiveInferenceAgent(state_size=STATE_DIM, action_size=ACTION_DIM, lr=1e-4, gamma=gamma, zeta=zeta)
env = NetworkEnvironment()

# Parametri di training
NUM_EPISODES = 100
MAX_STEPS_PER_EP = 80
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 50

wandb.init(
    project="active-inference-network",
    config={
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dim": HIDDEN_DIM,
        "gamma": gamma,
        "zeta": zeta,
        "lr": 1e-4,
        "episodes": NUM_EPISODES,
        "batch_size": BATCH_SIZE
    }
)

total_steps = 0
print("Starting Active Inference training...\n")

for episode in range(1, NUM_EPISODES + 1):
    obs, true_s = env.reset()
    last_action = np.random.randint(0, env.action_space_size)
    trajectory = [true_s.copy()]
    attack_flags = [env.attack]

    pos_losses, pol_losses, c_losses = [], [], []

    for step in range(MAX_STEPS_PER_EP):
        # --- Infer state nascosto ---
        inferred_state = agent.infer_state(obs, last_action)

        # --- Seleziona azione ---
        action = agent.choose_action(inferred_state)

        # --- Step ambiente ---
        next_obs, next_true_s, done, attack = env.step(action)
        trajectory.append(next_true_s.copy())
        attack_flags.append(attack)

        wandb.log({
            "step": total_steps,
            "packet_rate": next_true_s[0],
            "packet_loss": next_true_s[1],
            "latency": next_true_s[2],
            "action": action,
            "attack": int(attack)
        })

        # --- Memorizza esperienza (goal = 1 se attacco, 0 altrimenti) ---
        goal_signal = first_signals # packet_rate, loss, latency ideali -----------------------------------------------------------------------------------------------------
        agent.remember(true_s, last_action, action, next_true_s, obs, next_obs, goal_signal)

        # --- Allenamento su batch ---
        if len(agent.memory) >= BATCH_SIZE:
            p_loss, pol_loss, c_loss = agent.replay(BATCH_SIZE)
            if p_loss is not None:
                pos_losses.append(p_loss)
                pol_losses.append(pol_loss)
                c_losses.append(c_loss)

        # --- Aggiorna target network ---
        total_steps += 1
        if total_steps % TARGET_UPDATE_FREQ == 0:
            agent.update_target_critic()

        # --- Aggiorna stato ---
        obs, true_s = next_obs, next_true_s
        last_action = action

        # Termina episodio se l'azione "blocca IP" è stata presa
        if done:
            break

    # Report progress
    avg_pos = np.mean(pos_losses) if pos_losses else 0.0
    avg_pol = np.mean(pol_losses) if pol_losses else 0.0
    avg_c = np.mean(c_losses) if c_losses else 0.0
    attack_detected = attack_flags[-1]
    final_action = last_action

    print(f"Episode {episode:3d} | Steps: {len(trajectory) - 1:3d} | "
          f"Pos-loss: {avg_pos:6.4f} | Pol-loss: {avg_pol:6.4f} | C-loss: {avg_c:6.4f} | "
          f"Attack present: {'Yes' if attack_detected else 'No'} | Final action: {final_action} {'✅' if final_action == 2 else '❌'}"
          f"Attack steps: { env.current_step - env.attack_start_step if env.attack_started else "Nessun Attacco"}")

    wandb.log({
        "episode": episode,
        "posterior_loss": avg_pos,
        "policy_loss": avg_pol,
        "critic_loss": avg_c,
        "steps": len(trajectory) - 1,
        "attack_present": int(attack_detected),
        "final_action": final_action,
        "blocked": int(final_action == 2)
    })

torch.save(agent.policy_net.state_dict(), "policy.pth")
torch.save(agent.critic_net.state_dict(), "critic.pth")

wandb.save("policy.pth")
wandb.save("critic.pth")
print("\nTraining completed.")

# =============================================================================
# FINAL TEST EPISODE FOR NETWORK ENVIRONMENT
# =============================================================================
print("\n" + "=" * 60)
print("FINAL TEST EPISODE - NetworkEnvironment")
print("=" * 60)

# Reset environment
obs, true_s = env.reset()
trajectory_test = [true_s.copy()]
attack_flags_test = [env.attack]
last_action = 0

MAX_TEST_STEPS = 150

for step in range(MAX_TEST_STEPS):
    # --- Infer hidden state ---
    inferred_state = agent.infer_state(obs, last_action)

    # --- Compute EFE estimates for info ---
    state_tensor = torch.FloatTensor(inferred_state).unsqueeze(0).to(device)
    with torch.no_grad():
        efe_values = agent.critic_net(state_tensor)
        efe_values_np = efe_values.squeeze(0).cpu().numpy()

    # --- Choose action ---
    action = agent.choose_action(inferred_state)

    # --- Print debug info ---
    print(f"Step {step:3d}:")
    print(f"  Observed state: {obs}")
    print(f"  Inferred state: {inferred_state}")
    print(f"  EFE estimates: {efe_values_np}")
    print(
        f"  Chosen action: {action} {'(Reduce Noise)' if action == 0 else '(None)' if action == 1 else '(Block IP)'}\n")

    wandb.log({
        "test_step": step,
        "efe_action_0": efe_values_np[0],
        "efe_action_1": efe_values_np[1],
        "efe_action_2": efe_values_np[2],
        "chosen_action": action,
        "attack": int(attack)
    })

    # --- Step environment ---
    next_obs, next_true_s, done, attack = env.step(action)
    trajectory_test.append(next_true_s.copy())
    attack_flags_test.append(attack)

    # Update for next step
    obs, true_s = next_obs, next_true_s
    last_action = action

    if done:
        break

# --- Report results ---
trajectory_test = np.array(trajectory_test)
attack_detected = attack_flags_test[-1]

print(f"\nTest episode finished in {len(trajectory_test) - 1} steps")
print(f"Attack present in episode: {'Yes' if attack_detected else 'No'}")
print(f"Final action taken: {last_action} {'✅ BLOCK IP' if last_action == 2 else '❌ Not blocked'}")



# --- Optional: check if attack was mitigated ---
if attack_detected and last_action == 2:
    print("Attack successfully detected and blocked! 🎉")
elif attack_detected:
    print("Attack present but not blocked. ⚠️")
else:
    print("No attack in this episode. ✅")

# =============================================================================
# PLOT FINALE - 3 grafici separati in un'unica finestra
# =============================================================================

steps = np.arange(len(trajectory_test))

packet_rate = trajectory_test[:, 0]
packet_loss = trajectory_test[:, 1]
latency = trajectory_test[:, 2]

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# --- Packet Rate ---
axs[0].plot(steps, packet_rate)
for i, flag in enumerate(attack_flags_test):
    if flag:
        axs[0].axvspan(i, i+1, alpha=0.1)

axs[0].set_title("Packet Rate")
axs[0].set_ylabel("Rate")
axs[0].grid()


# --- Packet Loss ---
axs[1].plot(steps, packet_loss)
for i, flag in enumerate(attack_flags_test):
    if flag:
        axs[1].axvspan(i, i+1, alpha=0.1)

axs[1].set_title("Packet Loss")
axs[1].set_ylabel("%")
axs[1].grid()


# --- Latency ---
axs[2].plot(steps, latency)
for i, flag in enumerate(attack_flags_test):
    if flag:
        axs[2].axvspan(i, i+1, alpha=0.1)

axs[2].set_title("Latency")
axs[2].set_xlabel("Step")
axs[2].set_ylabel("ms")
axs[2].grid()

plt.tight_layout()
#plt.show()

wandb.log({
    "final_plot": wandb.Image(plt)
})

wandb.finish()
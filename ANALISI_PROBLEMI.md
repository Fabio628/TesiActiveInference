# Analisi tecnica del PoC: Active Inference per la cyber-difesa di rete

> **Destinatario:** studente di tesi  
> **Scopo:** spiegare in dettaglio i quattro problemi che impedivano la convergenza del codice originale, la teoria statistica sottostante, e tre direzioni di sviluppo che possono costituire il corpo della tesi.

---

## 1. Il problema dell'azione terminale — la politica degenere

### Cosa succedeva

Nel codice originale, l'azione `2` ("blocca IP") terminava immediatamente l'episodio:

```python
elif action == 2:  # termina episodio
    done = True
    obs = self._get_observation()
    return obs, self.state.copy(), done, self.attack
```

La funzione di costo che l'agente cercava di minimizzare era la distanza quadratica normalizzata dallo stato iniziale (la "zona omeostatica"):

```
penalità pragmatica = 0.5 · Σ_k  [(obs_k − μ*_k) / σ_k]²
```

All'istante `t = 0`, l'ambiente viene resettato esattamente su `state = [75, 2.5, 35]`, quindi `obs ≈ state` e la penalità è zero. Ad ogni passo successivo la penalità cresce (per le ragioni spiegate nella sezione 2).

Ne consegue che la **politica ottimale** per minimizzare la penalità futura è banalmente:

> **Bloccare immediatamente al passo 0**, prima che lo stato abbia avuto il tempo di allontanarsi dal baseline.

L'agente ha ragione dal punto di vista ottimizzatorio: stava risolvendo il problema che gli era stato assegnato. Il problema era la **formulazione**, non l'algoritmo.

### La soluzione adottata

Eliminare la terminazione anticipata. L'azione `2` diventa "mitiga" — un'azione che agisce sullo stato dell'ambiente ma non conclude l'episodio. Gli episodi durano sempre `MAX_STEPS = 100` passi. La penalità si accumula lungo tutta la traiettoria, rendendo l'agente obbligato a mantenere lo stato vicino al baseline per l'intera durata.

---

## 2. Il moto browniano — teoria statistica e conseguenze

### 2.1 Il random walk discreto

Il codice originale (versione `ActiveInferenceSurvival.py`) faceva evolvere lo stato così:

```python
drift = np.random.normal(0, [2, 0.5, 3])
self.state += drift
```

Questo è esattamente un **random walk discreto** (moto browniano discreto). Ogni incremento `ε_t ~ N(0, Σ)` è indipendente dai precedenti. Lo stato al tempo `t` è:

```
s_t = s_0 + Σ_{i=1}^{t} ε_i
```

Per proprietà delle variabili aleatorie indipendenti:

```
E[s_t] = s_0           (nessuna tendenza sistematica)
Var[s_t] = t · Σ²      (la varianza cresce linearmente col tempo!)
Std[s_t] = √t · σ      (la deviazione standard cresce come √t)
```

### 2.2 Il problema: la zona omeostatica è irraggiungibile

Con `σ = [2, 0.5, 3]` e un episodio di `T = 80` passi, la deviazione standard dello stato al termine dell'episodio è:

```
Std[s_80] = √80 · [2, 0.5, 3] ≈ [17.9, 4.5, 26.8]  (in unità assolute)
```

Normalizzando per `GOAL_STD = [10, 2, 10]`:

```
Std normalizzata[s_80] ≈ [1.79, 2.25, 2.68]  (in unità di deviazione standard)
```

Ciò significa che all'ottantesimo passo l'agente si trova, in media, a quasi **2 deviazioni standard** dalla zona omeostatica anche in assenza di qualsiasi attacco. Il costo pragmatico è inevitabilmente alto indipendentemente da qualsiasi azione intrapresa. L'agente non può "imparare" a restare vicino al baseline perché **il sistema fisico non ha nessuna forza di richiamo**: è un processo senza memoria.

### 2.3 La variante sinusoidale — perché non basta

La prima versione del codice (`ActiveInference.py`) sostituisce il random walk con un'oscillazione sinusoidale:

```python
self.phase += self.frequency
base_wave = np.sin(self.phase)
oscillation = np.array([10 * base_wave, 1.5 * base_wave, 8 * base_wave])
self.state += oscillation
```

Il segnale è deterministico e limitato, ma presenta due problemi:

1. **L'ampiezza è troppo grande.** Con ampiezza `10` e `GOAL_STD = 10`, l'oscillazione porta sistematicamente lo stato a `±1 σ` anche in assenza di attacchi. La zona omeostatica non è mai "calma".

2. **Richiede memoria temporale.** Per imparare che lo stato tornerà vicino al baseline dopo un massimo sinusoidale, l'agente dovrebbe capire *dove si trova nel ciclo*. Una rete feedforward senza memoria non può farlo: vede solo lo stato corrente, non la fase. Servirebbe una rete ricorrente (LSTM o simili).

### 2.4 La soluzione: il processo di Ornstein-Uhlenbeck

La correzione è sostituire il random walk con un **processo di Ornstein-Uhlenbeck (OU)**:

```
ds_t = θ (μ* − s_t) dt + σ dW_t
```

In forma discreta:

```python
self.state += theta * (GOAL_MEAN - self.state) + sigma * np.random.normal(0, 1, 3)
```

Il termine `θ(μ* − s_t)` è una **forza di richiamo** proporzionale alla distanza dal baseline. Quando lo stato si allontana da `μ*`, viene spinto verso di esso.

**Proprietà fondamentali:**

| Proprietà | Random walk | Processo OU |
|---|---|---|
| Media a lungo termine | `s_0` (dipende dall'inizio) | `μ*` (indipendente da `s_0`) |
| Varianza a lungo termine | `→ ∞` (diverge) | `σ²/(2θ)` (finita!) |
| Distribuzione stazionaria | Non esiste | `N(μ*, σ²/2θ)` |
| Memoria | Nessuna | Scala temporale `1/θ` |

La distribuzione stazionaria è una Gaussiana centrata sul baseline. Questo è esattamente il comportamento fisicamente plausibile di metriche di rete in condizioni normali: fluttuano intorno a un valore tipico, non derivano indefinitamente.

Con i parametri adottati (`θ = 0.25`, `σ = [1, 0.25, 2]`, `GOAL_STD = [5, 1, 10]`), la deviazione standard stazionaria del processo benign è:

```
σ_stazionaria = σ / √(2θ) = [1, 0.25, 2] / √0.5 ≈ [1.41, 0.35, 2.83]  (assoluta)
               → [0.28, 0.35, 0.28]  (normalizzata)
```

Le fluttuazioni benigne sono circa **0.3 σ** — piccole, contenute, con una zona omeostatica stabile.

---

## 3. Il segnale troppo debole — rapporto segnale/rumore

Anche con il processo OU, il codice iniziale aveva `ATK_RATE = [0.8, 0.2, 1.5]` per passo. Il punto di equilibrio dell'attacco si trova dove la forza di richiamo OU bilancia esattamente la deriva dell'attacco:

```
θ · (s_eq − μ*) = ATK_RATE
→ s_eq − μ* = ATK_RATE / θ = [3.2, 0.8, 6.0]  (assoluto)
→ deviazione normalizzata = [0.64, 0.80, 0.60] σ
```

La deviazione massima in condizioni di attacco sostenuto è **0.6–0.8 σ**, praticamente indistinguibile dalle normali fluttuazioni benigne (0.3 σ). Il **rapporto segnale/rumore** era circa **1.1×** — impossibile imparare da un segnale così debole.

La correzione fu aumentare `ATK_RATE` di circa 6×:

```python
ATK_RATE = np.array([5.0, 1.5, 10.0])
```

Il nuovo equilibrio sotto attacco:

```
deviazione normalizzata = [5/0.25, 1.5/0.25, 10/0.25] / [5, 1, 10] = [4.0, 6.0, 4.0] σ
```

L'attacco ora porta lo stato a **4–6 deviazioni standard** sopra il baseline. Il segnale è chiarissimo. Il differenziale EFE tra l'azione "mitiga" e "non fare nulla" durante un attacco è passato da `0.63` a circa `34` — cinquanta volte più grande.

---

## 4. L'input non normalizzato — geometria della funzione d'azione

La rete di policy e il critic ricevevano lo stato inferito *in unità assolute*: `[35, 2.5, 75]` in condizioni normali, `[55, 7, 105]` durante un attacco. Una rete neurale feedforward deve imparare una funzione del tipo "se il primo input è circa 55, il terzo circa 105, scegli azione 2". I pesi devono codificare la soglia assoluta, che dipende dall'ambiente specifico.

Dopo la normalizzazione:

```
input_normalizzato = (state − GOAL_MEAN) / GOAL_STD
```

In condizioni normali: `≈ [0, 0, 0]`. Sotto attacco: `≈ [4, 4, 3]`. La funzione da imparare diventa: "se i valori in input sono grandi e positivi, scegli azione 2". Questa è una funzione molto più semplice da rappresentare con una rete neurale piccola (`HIDDEN_DIM = 32`), ed è indipendente dall'unità di misura delle singole variabili.

---

## 5. Il termine epistemico: cosa è, perché è stato rimosso, e perché può costruire una tesi

### 5.1 Risposta diretta: sì, è stato rimosso

Il codice originale calcolava:

```python
epistemic_gain = -0.5 * torch.sum((true_s_t_plus_1 - inferred_s_t_plus_1) ** 2, dim=1)
one_step_efe = epistemic_gain + pragmatic_penalty
```

Nel `ActiveInferenceMinimal.py` questo termine **non è presente**. La EFE è puramente pragmatica:

```python
pragmatic = 0.5 * (norm_obs_dev ** 2).sum(dim=-1)
one_step_efe = pragmatic
```

### 5.2 Il problema del segno nel codice originale

Nella teoria dell'Inferenza Attiva, l'**Expected Free Energy** (EFE) si decompone in due termini:

```
G(π) = −E_π[log p̃(o)]          ← valore pragmatico  (distanza dalla zona omeostatica)
       − I_π[s ; o]              ← valore epistemico  (guadagno informativo atteso)
```

Il **valore epistemico** `I[s; o]` è la mutua informazione tra lo stato latente `s` e l'osservazione `o` sotto la politica `π`:

```
I_π[s; o] = KL[ q(s, o | π)  ||  q(s|π) · q(o|π) ]
           = E_q(o|π) [ KL[ q(s|o, π)  ||  q(s|π) ] ]
```

In parole: misura quanto un'osservazione futura ridurrebbe l'incertezza sullo stato latente. **Azioni che portano a osservazioni più informative hanno un valore epistemico più alto**, e quindi una EFE più bassa (ricordare che l'agente minimizza la EFE).

Il proxy usato dallo studente era `−||s_{t+1} − μ_{t+1}||²` (negativo dell'errore di ricostruzione). Interpretiamone il segno:

- Quando l'errore di ricostruzione è **grande** (il posterior è sbagliato sul prossimo stato): `epistemic_gain` diventa molto **negativo** → abbassa la EFE → l'azione viene **preferita**.
- Quando l'errore è **piccolo** (il posterior è accurato): `epistemic_gain ≈ 0` → la EFE è determinata solo dal termine pragmatico.

Il risultato è: **l'agente preferisce azioni che portano a stati in cui il posterior è sistematicamente errato**. Questo è il contrario di "ridurre l'incertezza" — è più simile a "cercare la sorpresa" o "evitare ciò che si conosce". Non è il comportamento epistemico desiderato.

Il termine corretto per il guadagno epistemico in forma approssimata (usata in letteratura, es. Millidge 2020) è:

```
I_π[s; o] ≈ H[q(s|π)] − E_q(o|π)[ H[q(s|o,π)] ]
```

cioè: entropia prior dello stato meno entropia media del posterior dopo l'osservazione. Per distribuzioni gaussiane questo si calcola come il rapporto tra varianza prior e varianza posterior. Il risultato è sempre non-negativo e misura effettivamente quanto l'osservazione aggiorna la credenza.

### 5.3 Perché graduale reintroduzione del termine epistemico può costruire una tesi

Questa è la parte più interessante. La sequenza di esperimenti che segue produce comportamenti qualitativamente diversi e osservabili, ognuno dei quali merita un'analisi propria:

#### Fase A — EFE puramente pragmatica (punto di partenza, già funzionante)

L'agente minimizza solo la distanza dalla zona omeostatica. Il comportamento emergente è una **politica a soglia**: se la deviazione normalizzata supera una soglia (appresa), mitiga; altrimenti aspetta. È equivalente a un detector statistico di anomalie con soglia adattiva.

**Metrica chiave:** `P(mitiga | attacco) / P(mitiga | benigno)` — il rapporto di selettività.

#### Fase B — Aggiunta del termine epistemico (proxy: varianza posterior)

Si aggiunge al critic un termine che premia le azioni che riducono l'incertezza del posterior:

```python
# Varianza stimata dell'errore posterior al passo successivo
epistemic_value = − 0.5 * ((s_t1 - mu_t1) / goal_std) ** 2  # segno CORRETTO
one_step_efe = pragmatic − lambda_ep * epistemic_value
```

(Attenzione: con questo segno, `epistemic_value` grande e positivo abbassa la EFE → azioni esplorative preferite.)

Il comportamento emergente: l'agente inizia a usare l'**azione 0 (monitor)** come precursore della mitigazione. Quando lo stato è moderatamente deviato (ambiguità tra attacco e fluttuazione benigna), l'azione "monitor" riduce il rumore di osservazione, consentendo una stima più precisa prima di decidere. Questo è il **comportamento di probing epistemico** — uno dei contributi teorici chiave dell'Inferenza Attiva rispetto al Reinforcement Learning.

**Metrica chiave:** sequenza temporale delle azioni — ci si aspetta di osservare pattern `monitor → mitiga` quando c'è incertezza.

#### Fase C — Termine epistemico corretto (mutua informazione)

Si sostituisce il proxy con una stima della mutua informazione vera:

- Il posterior net produce non solo una media `μ_t` ma anche una varianza `Σ_t` (trasformando PosteriorNet in una rete variazionale con output `(μ, log σ)`)
- L'entropia del prior predictive e del posterior vengono calcolate esplicitamente
- Il guadagno informativo è la loro differenza

Questo è il vero framework di Deep Active Inference (Millidge 2020, Mazzaglia et al. 2022).

**Comportamento atteso:** l'agente sviluppa comportamenti di esplorazione attiva più sofisticati — non solo riduzione del rumore, ma scelta di traiettorie che portano a osservazioni massimamente informative sul modo latente (attacco vs. benigno).

#### Schema progressivo della tesi

```
Capitolo 3: EFE pragmatica
   → dimostrare convergenza, analisi del comportamento a soglia

Capitolo 4: Aggiunta del termine epistemico (proxy)
   → osservare e misurare il probing behaviour
   → confronto con agente RL senza drive epistemico

Capitolo 5: Termine epistemico corretto (variazionale)
   → valutazione su scenari con alta ambiguità (attacchi lenti, flash crowd)
   → analisi del tradeoff pragmatico/epistemico al variare di λ_ep
```

Questo è esattamente il tipo di ablation study progressivo che rende una tesi sperimentale solida.

---

## 6. Il Posterior Net: stato continuo vs. credenza di attacco

### 6.1 Cosa fa attualmente il PosteriorNet

Il `PosteriorNet` prende in input l'osservazione rumorosa e l'azione precedente, e produce lo **stato inferito**:

```
PosteriorNet : (obs_t, a_{t-1}) → μ̂_t ∈ ℝ³
```

dove `ℝ³` è lo stesso spazio continuo delle metriche di rete: `[latenza, packet_loss, packet_rate]`. È addestrato a minimizzare `||μ̂_t − s_t||²`, cioè è un **denoiser**: impara a stimare lo stato vero a partire dall'osservazione rumorosa.

### 6.2 Cosa manca: la distribuzione di credenza sul modo latente

Nella teoria POMDP, lo stato nascosto `s_t` include non solo le metriche continue ma anche la **modalità operativa**: `s_t = (metriche_t, modalità_t)` con `modalità_t ∈ {benigno, attacco}`.

Attualmente, l'agente **non calcola esplicitamente** `P(attacco | osservazioni)`. La credenza che ci sia un attacco è implicita nella magnitude della deviazione normalizzata: quando `||(μ̂_t − GOAL_MEAN) / GOAL_STD||` è grande, la policy apprende statisticamente che conviene mitigare. Ma l'agente non ha mai un numero esplicito che rappresenta "sono al 73% sicuro che sia in corso un attacco".

Hai capito bene.

### 6.3 Come estendere al POMDP corretto

**Estensione 1 — Output ibrido del PosteriorNet:**

Modificare il PosteriorNet per produrre sia le metriche inferite che una probabilità di attacco:

```python
class PosteriorNet(nn.Module):
    def forward(self, obs, a_oh):
        h = self.shared_layers(torch.cat([obs, a_oh], dim=-1))
        metrics = self.head_metrics(h)   # ℝ³: stato continuo
        p_attack = torch.sigmoid(self.head_attack(h))  # ∈ (0,1): credenza attacco
        return metrics, p_attack
```

Il training richiederebbe un segnale supervisore per `p_attack` — in un sistema reale questo potrebbe venire da un IDS (Intrusion Detection System) esterno usato come "oracolo" durante l'addestramento.

**Estensione 2 — Processo OU con switch di regime (Hidden Markov Model):**

Il modo più formalmente corretto è usare un modello di transizione a due regimi:

```
P(modo_{t+1} = attacco  | modo_t = benigno) = p_attack       (5% per passo)
P(modo_{t+1} = benigno  | modo_t = attacco) = p_resolve      (50% se si mitiga)

s_{t+1} | modo_t = benigno ~ OU(μ*, σ_ben)
s_{t+1} | modo_t = attacco ~ OU(μ* + drift_atk, σ_atk)
```

La credenza del posterior si aggiorna con il filtro di Kalman nascosto (o con una variante neurale):

```
b_t(modo) = P(modo_t | o_{1:t}, a_{1:t})
```

Questo è esattamente il **belief state** del POMDP. L'agente mappa `b_t → azione`, che è la politica ottimale teorica. La parte di Inference Attiva aggiunge la minimizzazione della EFE come obiettivo di questa politica.

**Estensione 3 — pymdp per il caso discreto:**

La libreria `pymdp` (https://github.com/infer-actively/pymdp) implementa Inferenza Attiva discreta con le matrici canoniche:

- `A`: matrice di verosimiglianza `P(o | s)` — mappatura da stato a osservazione
- `B`: matrice di transizione `P(s_{t+1} | s_t, a)` — dinamica
- `C`: vettore di preferenze `p̃(o)` — la distribuzione preferita
- `D`: prior iniziale `p(s_0)`

Con stati discretizzati (es. 5 livelli di deviazione × 2 modalità = 10 stati), il belief update e il calcolo della EFE sono esatti e interpretabili. Questa è la versione più vicina alla teoria originale di Friston, e la più facile da validare pedagogicamente.

---

## Riepilogo dei problemi e delle correzioni

| # | Problema | Causa radice | Correzione |
|---|---|---|---|
| 1 | Politica degenere | Azione terminale dava penalità zero a t=0 | Sostituire con azione di mitigazione in-place |
| 2 | Stato divergente | Random walk / sinusoide senza forza di ripristino | Processo Ornstein-Uhlenbeck con μ* = GOAL_MEAN |
| 3 | Attacco indistinguibile | SNR ≈ 1.1× (attacco eq. ≈ rumore benigno) | Aumentare ATK_RATE fino a SNR > 5× |
| 4 | Input non centrato | Policy/critic su valori assoluti, non deviazioni | Normalizzare: `(s − μ*) / σ*` |
| 5 | Termine epistemico errato | Segno sbagliato — premiava errore, non informazione | Rimosso per ora; reintrodurre correttamente per la tesi |
| 6 | Posterior continuo | Nessuna credenza esplicita sul modo latente | Estensione futura: output `P(attacco)` o belief state discreto |

**Risultato dopo le correzioni:** dopo 400 episodi, l'agente raggiunge  
`P(mitiga | attacco) ≈ 0.46` vs `P(mitiga | benigno) ≈ 0.05` → rapporto di selettività **8.6×**. Converge.

---

*Ultima revisione: maggio 2026*

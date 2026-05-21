# Analisi tecnica del PoC: Active Inference per la cyber-difesa di rete

> **Scopo:** spiegare in dettaglio i quattro problemi che impedivano la convergenza del codice originale, la teoria statistica sottostante, e le direzioni di sviluppo che costituiscono il corpo di una tesi sull'argomento.

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

Una delle versioni del codice faceva evolvere lo stato così:

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

Ciò significa che all'ottantesimo passo ci si trova, in media, a quasi **2 deviazioni standard** dalla zona omeostatica anche in assenza di qualsiasi attacco. Il costo pragmatico è inevitabilmente alto indipendentemente da qualsiasi azione intrapresa. L'agente non può "imparare" a restare vicino al baseline perché **il sistema fisico non ha nessuna forza di richiamo**: è un processo senza memoria.

### 2.3 La variante sinusoidale — perché non basta

Un'altra versione del codice sostituisce il random walk con un'oscillazione sinusoidale:

```python
self.phase += self.frequency
base_wave = np.sin(self.phase)
oscillation = np.array([10 * base_wave, 1.5 * base_wave, 8 * base_wave])
self.state += oscillation
```

Il segnale è deterministico e limitato, ma presenta due problemi:

1. **L'ampiezza è troppo grande.** Con ampiezza `10` e `GOAL_STD = 10`, l'oscillazione porta sistematicamente lo stato a `±1 σ` anche in assenza di attacchi. La zona omeostatica non è mai "calma".

2. **Richiede memoria temporale.** Per imparare che lo stato tornerà vicino al baseline dopo un massimo sinusoidale, l'agente dovrebbe capire *dove si trova nel ciclo*. Una rete feedforward senza memoria non può farlo: vede solo lo stato corrente, non la fase. Servirebbe una rete ricorrente (LSTM o simili) — che sarebbe una direzione di ricerca interessante in sé.

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

Con i parametri adottati (`θ = 0.25`, `σ = [1, 0.25, 2]`, `GOAL_STD = [5, 1, 10]`), la deviazione standard stazionaria del processo benigno è:

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

La deviazione massima in condizioni di attacco sostenuto era **0.6–0.8 σ**, praticamente indistinguibile dalle normali fluttuazioni benigne (0.3 σ). Il **rapporto segnale/rumore** era circa **1.1×** — impossibile imparare da un segnale così debole.

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

### 5.1 Risposta diretta: sì, è stato rimosso dal PoC

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

### 5.2 Il termine epistemico nel codice: è corretto o no?

Questa è una domanda più sottile di quanto sembri a prima vista, e merita una risposta precisa.

Dalla derivazione completa della EFE (si veda la sezione 7 del tutorial di riferimento), la forma finale è:

```
EFE ≜ − E_{Q(ō|ā)} [ KL[ Q(s̄|ō,ā) ‖ q_ō(s̄|ā) ] ]   ← Guadagno Epistemico
      − E_{Q(ō|ā)} [ log Q(ō | C) ]                      ← Guadagno Pragmatico
```

Il termine epistemico è la **KL divergenza attesa** tra:
- `Q(s̄|ō,ā)`: il posterior *futuro* del modello generativo, cioè come cambieranno le credenze dell'agente dopo aver osservato le osservazioni future `ō`
- `q_ō(s̄|ā)`: il posterior *attuale* approssimato, cioè cosa crede l'agente ora

Questa KL misura **quanta informazione l'agente si aspetta di guadagnare** eseguendo l'azione `ā`. Poiché la EFE va minimizzata e il guadagno epistemico compare con segno negativo, **l'agente preferirà azioni che portano a osservazioni informative**, che aggiornano le sue credenze in modo significativo. È questa la "curiosità artificiale".

Sotto l'approssimazione gaussiana con varianza unitaria (standard nel Deep AIF, si veda il tutorial), la KL tra due gaussiane si riduce alla distanza L2 scalata:

```
KL[ N(μ₁, I) ‖ N(μ₂, I) ] = ½ ‖μ₁ − μ₂‖²
```

Il proxy nel codice è:

```python
epistemic_gain = -0.5 * torch.sum((true_s_t_plus_1 - inferred_s_t_plus_1) ** 2, dim=1)
```

Qui `true_s_t_plus_1` gioca il ruolo di `μ₁` (la media del posterior "vero", ottenibile dal replay buffer dove lo stato reale è memorizzato) e `inferred_s_t_plus_1` gioca il ruolo di `μ₂` (la media del posterior approssimato della rete). **Il segno è corretto**: il termine è negativo, quindi aggiunge un contributo negativo alla EFE, favorendo le azioni che producono alta incertezza residua nel posterior.

**L'interpretazione è la seguente**: l'agente preferisce azioni che conducono a stati in cui il suo modello interno commette errori di ricostruzione elevati. Questo è una forma valida di *curiosità diretta all'errore di modello* — esploro dove il mio modello è più sbagliato, perché lì imparerò di più. Questa interpretazione è diversa (ma non incompatibile) con la formulazione standard fristoniana, che confronta prior e posterior rispetto alle osservazioni future. Come osservato nel tutorial di riferimento, esistono *"otras escuelas de pensamiento igualmente válidas"* che adottano criteri leggermente diversi per il termine epistemico.

### 5.3 Allora perché è stato rimosso?

Il motivo è **pratico, non teorico**.

Nelle primissime fasi dell'addestramento il `PosteriorNet` ha pesi casuali: produce `inferred_s_t_plus_1` completamente arbitrario, indipendente da `true_s_t_plus_1`. Quindi `||true_s - inferred_s||²` è enorme e dominato dal rumore dell'inizializzazione casuale della rete, non dalla struttura dell'ambiente. Questo rende il gradiente della EFE instabile e impedisce al critic di apprendere la struttura pragmatica corretta prima che il termine epistemico abbia un senso.

In sintesi: il termine epistemico è teoricamente fondato, ma richiede che il posterior sia già sufficientemente addestrato per produrre un segnale utile. Nel PoC si è scelto di rimuoverlo per isolare il problema di convergenza pragmatica e validare la struttura di base.

### 5.4 Perché la reintroduzione graduale può costruire una tesi

Questa è la parte più interessante. La sequenza di esperimenti che segue produce comportamenti qualitativamente diversi e osservabili, ognuno dei quali merita un'analisi propria:

#### Fase A — EFE puramente pragmatica (punto di partenza, già funzionante)

L'agente minimizza solo la distanza dalla zona omeostatica. Il comportamento emergente è una **politica a soglia**: se la deviazione normalizzata supera una soglia (appresa), mitiga; altrimenti aspetta. È equivalente a un detector statistico di anomalie con soglia adattiva.

**Metrica chiave:** `P(mitiga | attacco) / P(mitiga | benigno)` — il rapporto di selettività.

#### Fase B — Aggiunta del termine epistemico (proxy dell'errore di modello)

Si aggiunge al critic il termine originale, ma con scalatura esplicita che bilancia i due contributi:

```python
norm_s_err  = (s_t1 - mu_t1) / goal_std   # normalizzato per confrontabilità
epistemic   = -0.5 * (norm_s_err ** 2).sum(dim=-1)

one_step_efe = pragmatic + lambda_ep * epistemic   # lambda_ep piccolo inizialmente
```

Il comportamento atteso: l'agente inizia a usare l'**azione 0 (monitor)** come precursore della mitigazione. Quando lo stato è moderatamente deviato (ambiguità tra attacco e fluttuazione benigna), "monitor" riduce il rumore di osservazione, consentendo una stima più precisa prima di decidere. Questo è il **probing behaviour epistemico** — uno dei contributi teorici chiave dell'Inferenza Attiva rispetto al Reinforcement Learning.

**Metrica chiave:** sequenza temporale delle azioni — ci si aspetta di osservare pattern `monitor → mitiga` quando c'è incertezza. Questo pattern è assente nell'agente puramente pragmatico.

**Iperparametro critico:** `lambda_ep` (il peso relativo del termine epistemico). Valori troppo alti rendono l'agente eccessivamente esplorativo e lento a mitigare attacchi certi; valori troppo bassi lo rendono indistinguibile dalla Fase A. Il tradeoff pragmatico/epistemico al variare di `lambda_ep` è un esperimento pulito e interpretabile.

#### Fase C — Termine epistemico corretto via posterior variazionale

Il proxy dell'errore di modello approssima la KL tra posterior vero e approssimato usando lo stato reale del replay buffer come sostituto del posterior vero. Una stima più rigorosa richiede che il `PosteriorNet` produca non solo la media `μ_t` ma anche una varianza `Σ_t` (trasformandolo in una rete variazionale con output `(μ, log σ)`):

```python
class PosteriorNet(nn.Module):
    def forward(self, obs, a_oh):
        h = self.shared(torch.cat([obs, a_oh], -1))
        mu    = self.head_mu(h)
        log_s = self.head_logs(h)       # log varianza
        return mu, log_s
```

Con distribuzione gaussiana piena, il guadagno epistemico diventa la KL analitica tra prior predictivo e posterior aggiornato — la stima più fedele alla formula di Friston. Il comportamento atteso è una esplorazione ancora più selettiva: l'agente distingue non solo "sono lontano dal baseline?" ma "sono incerto *perché* sono lontano?"

#### Schema progressivo della tesi

```
Capitolo 3: EFE pragmatica
   → dimostrare convergenza
   → analisi del comportamento a soglia, selettività P(mit|atk) vs P(mit|ben)

Capitolo 4: Aggiunta del termine epistemico (proxy errore di modello)
   → osservare e misurare il probing behaviour
   → confronto con agente RL senza drive epistemico (ablation study)
   → analisi del tradeoff pragmatico/epistemico al variare di λ_ep

Capitolo 5: Posterior variazionale e termine epistemico rigoroso
   → stima della KL via (μ, log σ) del posterior
   → analisi della discriminazione attacco/benigno con alta ambiguità
     (attacchi lenti, flash crowd, rumore non stazionario)
```

Questo è il tipo di ablation study progressivo che rende una tesi sperimentale solida: ogni capitolo aggiunge un solo grado di complessità e ne misura l'effetto su metriche già definite.

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

Attualmente, **non viene calcolata esplicitamente** `P(attacco | osservazioni)`. La credenza che ci sia un attacco è implicita nella magnitude della deviazione normalizzata: quando `||(μ̂_t − GOAL_MEAN) / GOAL_STD||` è grande, la policy apprende statisticamente che conviene mitigare. Ma l'agente non ha mai un numero esplicito che rappresenta "sono al 73% sicuro che sia in corso un attacco".

### 6.3 Come estendere al POMDP corretto

**Estensione 1 — Output ibrido del PosteriorNet:**

Modificare il PosteriorNet per produrre sia le metriche inferite che una probabilità di attacco:

```python
class PosteriorNet(nn.Module):
    def forward(self, obs, a_oh):
        h = self.shared_layers(torch.cat([obs, a_oh], dim=-1))
        metrics  = self.head_metrics(h)                  # ℝ³: stato continuo
        p_attack = torch.sigmoid(self.head_attack(h))   # ∈ (0,1): credenza attacco
        return metrics, p_attack
```

Il training richiederebbe un segnale supervisore per `p_attack` — in un sistema reale questo potrebbe venire da un IDS esterno usato come "oracolo" durante l'addestramento, oppure da un'architettura che inferisce la modalità in modo non supervisionato (es. GMM sul residuo di ricostruzione).

**Estensione 2 — Processo OU con switch di regime (Hidden Markov Model):**

Il modo più formalmente corretto è un modello di transizione a due regimi:

```
P(modo_{t+1} = attacco  | modo_t = benigno) = p_attack       (es. 4% per passo)
P(modo_{t+1} = benigno  | modo_t = attacco) = p_resolve      (es. 50% se si mitiga)

s_{t+1} | modo_t = benigno ~ OU(μ*, σ_ben)
s_{t+1} | modo_t = attacco ~ OU(μ* + drift_atk, σ_atk)
```

La credenza del posterior si aggiorna con il filtro bayesiano nascosto:

```
b_t(modo) = P(modo_t | o_{1:t}, a_{1:t})
```

Questo è esattamente il **belief state** del POMDP. L'agente mappa `b_t → azione`, che è la politica ottimale teorica. La parte di Inferenza Attiva aggiunge la minimizzazione della EFE come obiettivo di questa politica. L'implementazione in PyTorch via reti ricorrenti (LSTM o GRU) con stato latente distribuito è diretta e coerente con l'approccio Deep AIF con TD learning.

---

## Riepilogo dei problemi e delle correzioni

| # | Problema | Causa radice | Correzione |
|---|---|---|---|
| 1 | Politica degenere | Azione terminale dava penalità zero a t=0 | Sostituire con azione di mitigazione in-place |
| 2 | Stato divergente | Random walk / sinusoide senza forza di ripristino | Processo Ornstein-Uhlenbeck con μ* = GOAL_MEAN |
| 3 | Attacco indistinguibile | SNR ≈ 1.1× (attacco eq. ≈ rumore benigno) | Aumentare ATK_RATE fino a SNR > 5× |
| 4 | Input non centrato | Policy/critic su valori assoluti, non deviazioni | Normalizzare: `(s − μ*) / σ*` |
| 5 | Termine epistemico instabile | Posterior casuale in fase iniziale → segnale epistemico rumoroso domina pragmatico | Rimosso per ora; reintrodurre con scaling esplicito dopo convergenza del posterior |
| 6 | Posterior continuo | Nessuna credenza esplicita sul modo latente | Estensione futura: output `P(attacco)` o belief state con HMM in PyTorch |

**Risultato dopo le correzioni:** dopo 400 episodi, l'agente raggiunge  
`P(mitiga | attacco) ≈ 0.46` vs `P(mitiga | benigno) ≈ 0.05` → rapporto di selettività **8.6×**. Converge.

---

*Ultima revisione: maggio 2026*

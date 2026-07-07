# Clustering-based SSL for Action Discovery
- **Gruppo**: DataMinds (Lorenzo Comis, Alessandro Sciacca)
- **Progetto**: Track 12 — Self-Supervised Learning

---

## 1. Introduzione e obiettivo

Il progetto affronta la **scoperta non supervisionata di azioni** in video: dato un insieme di clip senza alcuna annotazione, l'obiettivo è far emergere raggruppamenti che corrispondano ad azioni ricorrenti, usando le etichette reali **esclusivamente come chiave di valutazione a posteriori** — mai durante il training. La traccia (Track 12) fissa quattro obiettivi minimi: una baseline K-Means su feature ResNet18, l'estrazione di feature con un modello self-supervised (VideoMAE), un **K-Means iterativo** in cui il clustering e la rappresentazione si migliorano a vicenda, e la valutazione tramite corrispondenza con le classi reali.

La domanda scientifica centrale che ci siamo posti è questa: *il fine-tuning iterativo guidato da pseudo-label può rendere clusterizzabili delle rappresentazioni che non nascono per esserlo?* Le feature di ResNet18 (supervisionate su ImageNet) partono già semanticamente organizzate; quelle di VideoMAE (pre-training di ricostruzione mascherata) no — e il divario di partenza tra le due è enorme. Il clustering iterativo in stile DeepCluster promette di colmare questo tipo di divario, ma è nato per milioni di immagini: capire **dove funziona, dove no, e perché** su un problema di scala accademica è stato il filo conduttore del lavoro.

Tre principi hanno guidato tutto il progetto, e questo report li riflette: (i) le aspettative sono state trattate come **ipotesi da verificare**, non come risultati da confermare — e i risultati negativi sono documentati con lo stesso rigore di quelli positivi; (ii) le etichette reali sono confinate alla valutazione **per costruzione del codice**, non per sola disciplina; (iii) la **riproducibilità** è un requisito: ogni numero di questo report è rigenerabile da un file di configurazione e un seed, e la cronologia git del repository costituisce il registro fedele e verificabile dell'intero percorso, nell'ordine in cui è realmente avvenuto.

## 2. Contributi e valore aggiunto

Oltre all'implementazione completa della pipeline richiesta dalla traccia, il progetto porta questi contributi:

1. **Un'implementazione del clustering iterativo con garanzie strutturali**: criterio di stop interno basato sulla stabilità delle pseudo-label (NMI tra assegnazioni consecutive), etichette reali architetturalmente fuori dal flusso di controllo, contromisure al collasso dei cluster (cross-entropy pesata, testa reinizializzata a ogni iterazione).
2. **Uno studio di scaling a tre punti** (398 → 1.865 → 5.802 video), reso possibile dall'estensione autonoma del dataset dagli archivi ufficiali di Kinetics-400: è l'esperimento che ha trasformato un risultato negativo in una mappa del comportamento del metodo.
3. **Un risultato di calibrazione con valore generale**: il regime di fine-tuning va scalato con la dimensione del dataset (in numero di update, non di epoche) — dimostrato con controfattuali su entrambi i backbone.
4. **Una scoperta collaterale a costo zero**: la sola L2-normalization delle feature prima del K-Means vale **+8,5 punti di purity** sulla baseline ResNet.
5. **Due contributi di rigore** riusabili oltre questo progetto: la diagnosi di un difetto di caricamento dei pesi pre-addestrati di VideoMAE nelle versioni recenti di `transformers` (con test di guardia permanente), e una pipeline verificata come **bit-deterministica** sia su CPU sia su GPU.
6. **Ablation documentate** (over-clustering K=50, capacità del backbone) che validano la configurazione adottata per misura e non per scelta arbitraria.

## 3. Dati utilizzati

Il punto di partenza è un sottoinsieme di **Kinetics-400** con 10 classi di azione (*archery, driving car, javelin throw, passing American football, playing drums, playing guitar, playing tennis, pull ups, scuba diving, squat*): **398 video** (~35–45 per classe), organizzati come `data/<classe>/*.mp4`. Da ogni video vengono campionati **16 frame uniformi**, ridimensionati a 224×224.

Il dataset è poi cresciuto due volte nel corso del progetto, e le ragioni sono parte integrante della storia sperimentale (§5). Quando i primi esperimenti hanno indicato che 398 campioni erano il probabile collo di bottiglia del metodo, abbiamo esteso il dataset attingendo agli **archivi tar ufficiali** di Kinetics-400 (CVD Foundation) con uno script di harvesting in streaming scritto ad hoc: ogni archivio viene scaricato, filtrato tramite i CSV ufficiali di annotazione tenendo solo i video delle nostre 10 classi, ed eliminato — così il picco di occupazione disco resta di ~2 GB a fronte di centinaia di GB di traffico. La prima estensione (split *val* e *test*) ha portato il dataset a **1.865 video**; la seconda (split *train*, con arresto automatico al raggiungimento di 500 video per classe) a **5.802 video**, dopo la rimozione di 6 file corrotti individuati come clip interamente nere in fase di decodifica.

**Tabella 1 — Le tre fasi del dataset**

| Fase | Video totali | Per classe | Provenienza |
| :--- | :---: | :---: | :--- |
| Iniziale | 398 | 34–45 | selezione dallo split *train* |
| Prima estensione | 1.865 | 175–195 | + split *val* e *test* ufficiali |
| Seconda estensione | 5.802 | 500–650 | + split *train* (stop a quota 500/classe) |

![Crescita del dataset](../figures/dataset_growth.png)

Due note metodologiche. **(a)** Mescolare gli split di Kinetics è legittimo nel nostro contesto: non esiste un addestramento supervisionato con train/test da separare — le etichette servono solo a valutare, e gli split sono per noi semplici contenitori di video. La deduplicazione è garantita per costruzione (i 398 video originali provengono dallo split *train*; verificato: zero sovrapposizioni con le estensioni). **(b)** Come preprocessing, i frame decodificati vengono cachati su disco in `uint8` una sola volta; la **data augmentation** (flip orizzontale e random resized crop, campionati una volta per clip e applicati identici a tutti i 16 frame per preservare la coerenza temporale) è applicata solo durante il fine-tuning, mai in fase di estrazione feature.

## 4. Metodologia e architettura

### 4.1 Estrazione delle feature e baseline

Due backbone estraggono una rappresentazione per video: **ResNet18** pre-addestrata su ImageNet (forward sui 16 frame, mean pooling temporale → 512 dimensioni) e **VideoMAE-base** pre-addestrato con masked autoencoding su video (mean pooling dei token → 768 dimensioni; `transformers` è fissato alla serie 4.x dopo aver scoperto — e verificato con un test permanente — che le versioni 5.x scartano silenziosamente i bias di attention pre-addestrati del checkpoint). La baseline dell'obiettivo 1 è un singolo K-Means con K=10 (pari al numero di classi, assunzione dichiarata) sulle feature così estratte.

La valutazione usa tre metriche complementari, calcolate da un unico modulo condiviso tra baseline e metodo iterativo: **purity** (frazione di campioni coerenti con la classe maggioritaria del proprio cluster — coincide con l'accuracy majority-vote, ma cresce meccanicamente con K), **NMI** (informazione mutua normalizzata, robusta al numero di cluster) e **ARI** (indice di Rand corretto per il caso, severo verso i cluster spezzati). Nessuna metrica da sola basta; insieme triangolano.

### 4.2 Il loop iterativo

![Schema del loop iterativo](../figures/iterative_loop_schema.png)

Il metodo segue lo schema DeepCluster: le assegnazioni del K-Means corrente diventano **pseudo-label** per un breve fine-tuning del backbone; con il backbone aggiornato si riestraggono le feature e si ri-clusterizza; e così via. L'**iterazione 0** — K-Means sulle feature del modello pre-addestrato, prima di qualunque fine-tuning — coincide per costruzione con la baseline, il che rende ogni curva direttamente confrontabile col punto di partenza. Le scelte qualificanti:

- **Clustering**: K-Means su feature **L2-normalizzate** (equivalente a una geometria coseno, più adatta a feature di reti profonde; l'effetto è quantificato in §5.1), K=10 fisso.
- **Fine-tuning selettivo**: con un dataset piccolo e pseudo-label rumorose, adattare tutto il backbone distruggerebbe il pre-training. Si sbloccano solo i layer semantici alti: `layer4` per ResNet18 (8,4M parametri su 11,2M, con statistiche BatchNorm congelate — i batch piccoli di frame correlati le stimerebbero male) e gli **ultimi 2 blocchi** encoder per VideoMAE (14,2M su 86,2M).
- **Testa e ottimizzatore ricreati a ogni iterazione**: gli ID dei cluster permutano arbitrariamente tra un'iterazione e l'altra, quindi riusare la testa non ha senso; e i momenti di AdamW accumulati su pseudo-label vecchie non ne hanno su quelle nuove. Learning rate differenziati (basso per il backbone pre-addestrato, alto per la testa appena inizializzata).
- **Anti-collasso**: cross-entropy **pesata con l'inverso della dimensione dei cluster** — senza, i cluster grandi dominano il gradiente e vengono rinforzati a ogni giro fino a degenerare. Le dimensioni dei cluster sono monitorate a ogni iterazione come "canarino" del collasso.
- **Criterio di stop interno**: il loop si ferma quando la NMI **tra le assegnazioni di due iterazioni consecutive** supera 0,95 (le pseudo-label si sono stabilizzate), con un tetto massimo di iterazioni. Questo confronto coinvolge solo coppie di partizioni non supervisionate: le etichette reali non partecipano alla decisione. Nel codice la separazione è **strutturale** — la funzione che valuta purity/NMI/ARI è l'unico punto di contatto con le etichette, viene chiamata a valle delle decisioni e il suo output finisce solo nei log. Per lo stesso principio, il risultato dichiarato di ogni run è l'**ultima iterazione** (scelta dal criterio interno), mai la migliore secondo le metriche vere.

### 4.3 Infrastruttura di calcolo

Il progetto è iniziato interamente sul portatile di uno di noi (solo CPU). Lì sono nate la pipeline, le verifiche e i primi run ResNet (~2 ore l'uno); ma per VideoMAE un singolo run iterativo completo avrebbe richiesto **~15 ore stimate**, e la campagna sperimentale ne prevedeva molti. Siamo quindi passati al **cluster GPU del DMI** (SLURM + Apptainer, GPU NVIDIA L40S), adattando la pipeline ai suoi vincoli reali: nodi senza accesso a Internet (pacchetti installati offline da wheel pre-scaricate, modelli HuggingFace copiati nella cache del cluster), un job per utente alla volta, risorse dichiarate in anticipo. Il determinismo dell'intera pipeline (due esecuzioni identiche → storici, assegnazioni e pesi finali identici al bit) è verificato da script dedicati in `tests/` ed è stato ri-verificato sul nuovo hardware prima di qualunque esperimento.

**Tabella 2 — Tempi misurati, portatile vs cluster**

| Operazione | Portatile (CPU) | Cluster (L40S) |
| :--- | :---: | :---: |
| Run iterativo ResNet, 398 video | ~2 h | ~8 min |
| Run iterativo VideoMAE, 398 video | ~15 h (stima) | 9 min |
| Run iterativo VideoMAE, 1.865 video | impraticabile | 22 min |
| Run iterativo VideoMAE, 5.802 video (15 iter.) | impraticabile | ~2 h |

Senza il cluster, lo studio di scaling di §5 non sarebbe semplicemente esistito: la seconda metà della campagna (7 run su dataset estesi, più le ablation) è costata in totale meno di una giornata di calcolo.

---

## 5. Risultati e discussione

Presentiamo gli esperimenti nell'ordine in cui sono avvenuti, perché ogni passo è motivato dall'esito del precedente.

### 5.1 Baseline, e una scoperta a costo zero

**Tabella 3 — Baseline sul dataset iniziale (398 video, K=10)**

| Configurazione | Purity | NMI | ARI |
| :--- | :---: | :---: | :---: |
| ResNet18, K-Means | 0.5050 | 0.4898 | 0.3024 |
| ResNet18, K-Means + **L2-norm** | **0.5905** | **0.5571** | **0.4013** |
| VideoMAE, K-Means | 0.3367 | 0.2642 | 0.1176 |
| VideoMAE, K-Means + L2-norm | 0.3417 | 0.2642 | 0.1157 |

Due fatti saltano all'occhio. Primo: la sola **L2-normalization** delle feature prima del K-Means vale **+8,5 punti di purity** su ResNet — passare alla geometria coseno riorganizza gratis uno spazio già semanticamente strutturato (su VideoMAE, dove la struttura non c'è, non cambia quasi nulla). Secondo: il **divario tra i due backbone è di ~25 punti**. Non è un difetto di implementazione (lo abbiamo verificato fin nei pesi del checkpoint): il pre-training di ricostruzione mascherata ottimizza il modello a *ricostruire*, non a *separare* — le feature MAE grezze sono note per richiedere un adattamento supervisionato prima di diventare discriminative. Ridurre questo divario senza etichette è esattamente la sfida dell'obiettivo 3.

### 5.2 ResNet iterativo: il regime di training decide tutto

![ResNet 398: regime aggressivo vs gentile](../figures/resnet398_regimes.png)

Il primo run del loop (2 epoche/iterazione, lr backbone 10⁻⁴) è stato un fallimento istruttivo: la loss crollava sotto 0,5 già in due epoche — la rete *memorizzava* le pseudo-label — e la partizione si rimescolava a ogni giro senza migliorare (purity finale sotto la baseline). La diagnosi ci ha dato il termometro che abbiamo usato per tutto il resto del progetto: **la loss di training misura l'aggressività del regime**. Con passi dieci volte più piccoli (1 epoca, lr 10⁻⁵) lo stesso identico loop si è messo a *raffinare*: purity da 0.5905 a **0.6206** (+3,0), ARI +2,1, con la stabilità delle pseudo-label in crescita monotona. Regola empirica emersa: il miglioramento deve venire da tante iterazioni brevi, non da poche lunghe.

### 5.3 VideoMAE a 398 video: un risultato negativo onesto

Sullo stesso dataset, il loop calibrato non ha smosso VideoMAE: +0,7 punti di purity, stabilità ferma tra 0,71 e 0,85 per 15 iterazioni senza mai convergere. Il training funzionava (loss in discesa regolare), i cluster non collassavano: semplicemente **il segnale non bastava a innescare il bootstrap** su feature prive di struttura semantica di partenza. L'ipotesi più fondata era la scala: DeepCluster, il riferimento del metodo, opera su 1,3 milioni di immagini — noi ne avevamo 398. Da qui la prima estensione del dataset (§3).

### 5.4 Lo studio di scaling: il metodo si innesca, poi satura

![VideoMAE alle tre scale](../figures/videomae_scaling.png)

**Tabella 4 — VideoMAE alle tre scale (K=10, 2 blocchi; Δ = finale − iterazione 0)**

| Scala | Iter. 0 (P/NMI/ARI) | Finale (P/NMI/ARI) | Δ (punti) | Stop |
| :--- | :---: | :---: | :---: | :--- |
| 398 | 0.342 / 0.264 / 0.116 | 0.349 / 0.283 / 0.129 | +0.7 / +1.9 / +1.4 | tetto (15) |
| 1.865 | 0.301 / 0.226 / 0.111 | 0.335 / 0.252 / 0.145 | **+3.3 / +2.6 / +3.4** | **convergenza (8)** |
| 5.802 (LR/3) | 0.311 / 0.221 / 0.119 | 0.319 / 0.241 / 0.135 | +0.8 / +2.0 / +1.5 | convergenza (7) |

**Tabella 5 — ResNet18 alle tre scale (regime gentile; Δ = finale − iterazione 0)**

| Scala | Iter. 0 (P/NMI/ARI) | Finale (P/NMI/ARI) | Δ (punti) | Stop |
| :--- | :---: | :---: | :---: | :--- |
| 398 | 0.591 / 0.557 / 0.401 | 0.621 / 0.554 / 0.422 | +3.0 / −0.3 / +2.1 | tetto (10) |
| 1.865 | 0.597 / 0.533 / 0.381 | 0.600 / 0.542 / 0.416 | +0.2 / +0.9 / +3.5 | convergenza (8) |
| 5.802 (LR/3) | 0.596 / 0.521 / 0.385 | 0.607 / 0.532 / 0.422 | +1.2 / +1.1 / +3.6 | convergenza (9) |

![Guadagni per scala](../figures/scaling_deltas.png)

![Convergenza del loop](../figures/stability_convergence.png)

A **1.865 video** è arrivata la svolta: per la prima volta il loop su VideoMAE è **convergito autonomamente** (stabilità ≥ 0,95 all'iterazione 8) migliorando tutte le metriche — l'ARI, la più severa, del 30% relativo. Il confronto corretto è sempre *dentro* la stessa scala (le baseline assolute non sono confrontabili tra scale: i video aggiunti rendono il task più vario), e dentro la scala il salto 398→1.865 è netto su ogni metrica.

Il terzo punto ha però richiesto un passaggio in più, che è diventato un risultato a sé. Rieseguendo a **5.802 video** con la stessa configurazione, *entrambi* i backbone sono regrediti (ResNet: Δ purity −1,0) con la loss di nuovo in zona memorizzazione: "1 epoca" a scala tripla significa il triplo dei passi di gradiente per iterazione — **la gentilezza del regime va misurata in update, non in epoche**, e non ricalibrarla riporta la patologia di §5.2. Riducendo il learning rate del backbone dello stesso fattore di crescita del dataset (÷3), la convergenza è tornata su entrambi i backbone e ResNet è tornato positivo: il run non calibrato resta nel registro come controfattuale che isola l'effetto. Con il regime a posto, il verdetto sul terzo punto è pulito: **il beneficio del loop su VideoMAE ha il suo massimo attorno a ~2.000 video e poi satura** (+0,8 di purity a 5.802 contro +3,3 a 1.865, con convergenza ancora più rapida). Più dati accendono il metodo, ma non lo fanno crescere indefinitamente.

### 5.5 Ablation: la configurazione standard, validata per misura

![Ablation su VideoMAE](../figures/ablations_videomae.png)

Sulla scala intermedia abbiamo testato le due leve "gratuite" che avrebbero potuto spingere VideoMAE più su. **Over-clustering (K=50)**, l'ingrediente canonico di DeepCluster: da noi frammenta senza guadagnare — guadagni interni dimezzati, nessuna convergenza, cluster degeneri da 3–4 campioni; con ~37 video per cluster attesi, i sotto-cluster non hanno la popolosità che rende l'over-clustering utile a scala DeepCluster. **Più capacità (4 blocchi invece di 2)**: converge prima ma a una partizione peggiore (Δ quasi dimezzati), con la loss più bassa a fare da firma — la capacità extra si spende in memorizzazione delle pseudo-label, non in struttura. È la stessa lezione del regime di §5.2, vista dal lato dei parametri. Entrambe le ablation sono negative, ed è il loro valore: la configurazione standard (K=10, 2 blocchi, regime gentile scalato) non è una scelta arbitraria ma un massimo locale misurato.

### 5.6 Lettura d'insieme

La mappa che emerge dai sei esperimenti: su **feature mature** (ResNet/ImageNet) il loop iterativo *raffina* — la purity ha un tetto vicino alla baseline L2 (~0,60–0,62, invariante alla scala) ma la coerenza strutturale della partizione migliora sempre (ARI +2÷3,6 a ogni scala); su **feature grezze** (VideoMAE) il loop *costruisce*, ma solo sopra una scala minima di dati che ne innesca il bootstrap, e con rendimenti che saturano presto. Il divario tra i due mondi (~0,32 vs ~0,61 di purity) si riduce ma non si colma: alle scale accessibili a un progetto accademico, il segnale delle pseudo-label non sostituisce la supervisione — la quantifica, ed è un'informazione.

Una nota qualitativa dalle assegnazioni: le classi con firma visiva di scena forte (es. *scuba diving*, dominata dal blu subacqueo) formano cluster puri già in baseline, mentre azioni diverse in ambienti simili (es. *pull ups* e *squat*, entrambe in palestra) restano le più confuse — coerente con backbone che vedono più la scena che il movimento, dato anche il mean pooling temporale che scarta la dinamica.

## 6. Conclusioni e limiti

Tutti e quattro gli obiettivi minimi della traccia sono raggiunti, e la domanda centrale ha una risposta articolata ma netta: **il clustering iterativo funziona — converge autonomamente e migliora le proprie partizioni su entrambi i backbone — ma il suo effetto dipende criticamente da tre condizioni misurate**: la maturità delle feature di partenza (decide *cosa* il loop può fare: raffinare vs costruire), la scala dei dati (decide *se* si innesca) e la calibrazione del regime di training (decide se raffina o memorizza).

I limiti, dichiarati: **(i)** K=10 assume noto il numero di azioni — realistico per la valutazione, generoso per lo scenario "discovery" puro; **(ii)** anche il nostro dataset esteso resta 2–3 ordini di grandezza sotto la scala nativa di DeepCluster, e la saturazione osservata potrebbe non essere l'ultima parola a scale molto maggiori; **(iii)** il mean pooling temporale scarta la dinamica del movimento, probabilmente penalizzando proprio le classi che più ne avrebbero bisogno; **(iv)** l'augmentation è solo spaziale. Con più tempo, le direzioni che riteniamo più promettenti sono l'aggiunta di un segnale contrastivo o temporale accanto alle pseudo-label (per dare al bootstrap un appiglio indipendente dal clustering), pooling temporali più espressivi, e la stima non supervisionata di K.

## 7. Informazioni aggiuntive

### 7.1 Suddivisione dei contributi

- **Alessandro Sciacca**: caricamento del dataset e campionamento dei frame, estrattori di feature ResNet18 e VideoMAE, pipeline di estrazione con caching, baseline K-Means.
- **Lorenzo Comis**: clustering iterativo completo (loop, fine-tuning selettivo, criterio di stop), modulo di valutazione (purity/NMI/ARI), estensioni del dataset, campagna sperimentale in locale e sul cluster GPU, verifiche di riproducibilità e determinismo, figure e report.

Le decisioni di metodo e l'interpretazione dei risultati sono state discusse congiuntamente nel gruppo.

### 7.2 Uso dell'intelligenza artificiale

Dichiariamo l'uso di strumenti di AI generativa nel progetto, nei termini seguenti — verificabili rispetto alla cronologia git del repository. La **direzione scientifica, le decisioni di metodo e di architettura, la formulazione delle ipotesi, l'interpretazione dei risultati e i controlli di rigore** sono opera nostra. Abbiamo usato l'AI generativa come **supporto all'implementazione, sotto la nostra guida**: parte del codice della pipeline, gli script di test e verifica, gli script di orchestrazione per il cluster SLURM e la revisione stilistica di questo report. Le idee, i ragionamenti e le osservazioni che questo report descrive sono sempre stati nostri: a ogni passo abbiamo valutato le opzioni proposte, deciso la strada e verificato i risultati.

---

*I comandi per riprodurre ogni esperimento (setup dell'ambiente, download del dataset, run e verifiche) sono documentati nel `README.md` del repository.*

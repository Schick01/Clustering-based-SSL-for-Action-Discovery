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

Due backbone estraggono una rappresentazione per video: **ResNet18** pre-addestrata su ImageNet (forward sui 16 frame, mean pooling temporale → 512 dimensioni) e **VideoMAE-base** pre-addestrato con masked autoencoding su video (mean pooling dei token → 768 dimensioni). La baseline dell'obiettivo 1 è un singolo K-Means con K=10 (pari al numero di classi, assunzione dichiarata) sulle feature così estratte.

La valutazione usa tre metriche complementari, calcolate da un unico modulo condiviso tra baseline e metodo iterativo: **purity** (frazione di campioni coerenti con la classe maggioritaria del proprio cluster — coincide con l'accuracy majority-vote, ma cresce meccanicamente con K), **NMI** (informazione mutua normalizzata, robusta al numero di cluster) e **ARI** (indice di Rand corretto per il caso, severo verso i cluster spezzati). Nessuna metrica da sola basta; insieme triangolano.

### 4.2 Il loop iterativo

![Schema del loop iterativo](../figures/iterative_loop_schema.png)

Il metodo segue lo schema DeepCluster: le assegnazioni del K-Means corrente diventano **pseudo-label** per un breve fine-tuning del backbone; con il backbone aggiornato si riestraggono le feature e si ri-clusterizza; e così via. L'**iterazione 0** — K-Means sulle feature del modello pre-addestrato, prima di qualunque fine-tuning — coincide per costruzione con la baseline, il che rende ogni curva direttamente confrontabile col punto di partenza. Le scelte qualificanti:

- **Clustering**: K-Means su feature **L2-normalizzate** (equivalente a una geometria coseno, più adatta a feature di reti profonde; l'effetto è quantificato in §5.1), K=10 fisso.
- **Fine-tuning selettivo**: con un dataset piccolo e pseudo-label rumorose, adattare tutto il backbone distruggerebbe il pre-training. Si sbloccano solo i layer semantici alti: `layer4` per ResNet18 (8,4M parametri su 11,2M, con statistiche BatchNorm congelate — i batch piccoli di frame correlati le stimerebbero male) e gli **ultimi 2 blocchi** encoder per VideoMAE (14,2M su 86,2M).
- **Testa e ottimizzatore ricreati a ogni iterazione**: gli ID dei cluster permutano arbitrariamente tra un'iterazione e l'altra, quindi riusare la testa non ha senso; e i momenti di AdamW accumulati su pseudo-label vecchie non ne hanno su quelle nuove. Learning rate differenziati (basso per il backbone pre-addestrato, alto per la testa appena inizializzata).
- **Anti-collasso**: cross-entropy **pesata con l'inverso della dimensione dei cluster** — senza, i cluster grandi dominano il gradiente e vengono rinforzati a ogni giro fino a degenerare. Le dimensioni dei cluster sono monitorate a ogni iterazione come "canarino" del collasso.
- **Criterio di stop interno**: il loop si ferma quando la NMI **tra le assegnazioni di due iterazioni consecutive** supera 0,95 (le pseudo-label si sono stabilizzate), con un tetto massimo di iterazioni. Questo confronto coinvolge solo coppie di partizioni non supervisionate: le etichette reali non partecipano alla decisione. Nel codice la separazione è **strutturale** — la funzione che valuta purity/NMI/ARI è l'unico punto di contatto con le etichette, viene chiamata a valle delle decisioni e il suo output finisce solo nei log. Per lo stesso principio, il risultato dichiarato di ogni run è l'**ultima iterazione** (scelta dal criterio interno), mai la migliore secondo le metriche vere.

### 4.3 Riproducibilità: i tre cancelli

Prima di produrre qualunque risultato abbiamo imposto tre verifiche bloccanti ("cancelli"), ciascuna con uno script dedicato in `tests/`:

**Tabella 2 — I cancelli di rigore**

| Cancello | Cosa verifica | Esito |
| :--- | :--- | :--- |
| Fedeltà della cache | le feature estratte dalla cache dei frame sono identiche a quelle estratte dai video originali, per entrambi i backbone | ResNet bit-identico; VideoMAE ≤ 2·10⁻⁶ |
| Pesi pre-addestrati | i bias di attention di VideoMAE coincidono col checkpoint ufficiale su tutti i 12 layer | superato (dopo pin di `transformers`) |
| Determinismo | due esecuzioni identiche della CLI producono storici, assegnazioni e checkpoint identici | bit-identico, su CPU **e** su GPU |

Il secondo cancello nasce da una scoperta non banale: le versioni 5.x di `transformers` **scartano silenziosamente i bias di attention pre-addestrati** di VideoMAE (formato `q_bias`/`v_bias` del checkpoint non più mappato), reinizializzandoli a zero. Abbiamo quantificato l'impatto rieseguendo la baseline con i pesi corretti: le feature cambiano dell'8–15% in norma ma le metriche di clustering si spostano di meno di un punto — la baseline debole di VideoMAE è quindi una proprietà reale, non un artefatto. Abbiamo comunque fissato `transformers==4.57.1` (con un test di guardia permanente), perché fine-tunare un modello privo dei bias su 10 blocchi congelati non sarebbe stato difendibile. Il terzo cancello ha invece rivelato, al primo tentativo su GPU, il **non-determinismo del backward dei kernel fused di attention**: risolto forzando algoritmi deterministici e il percorso "math" per l'attention — da allora qualunque operazione non deterministica solleva un errore esplicito invece di divergere in silenzio.

### 4.4 Infrastruttura di calcolo

Il progetto è iniziato interamente sul portatile di uno di noi (solo CPU). Lì sono nate la pipeline, le verifiche e i primi run ResNet (~2 ore l'uno); ma per VideoMAE un singolo run iterativo completo avrebbe richiesto **~15 ore stimate**, e la campagna sperimentale ne prevedeva molti. Siamo quindi passati al **cluster GPU del DMI** (SLURM + Apptainer, GPU NVIDIA L40S), adattando la pipeline ai suoi vincoli reali: nodi senza accesso a Internet (pacchetti installati offline da wheel pre-scaricate, modelli HuggingFace copiati nella cache del cluster), un job per utente alla volta, risorse dichiarate in anticipo. Il determinismo è stato ri-verificato sul nuovo hardware prima di qualunque esperimento.

**Tabella 3 — Tempi misurati, portatile vs cluster**

| Operazione | Portatile (CPU) | Cluster (L40S) |
| :--- | :---: | :---: |
| Run iterativo ResNet, 398 video | ~2 h | ~8 min |
| Run iterativo VideoMAE, 398 video | ~15 h (stima) | 9 min |
| Run iterativo VideoMAE, 1.865 video | impraticabile | 22 min |
| Run iterativo VideoMAE, 5.802 video (15 iter.) | impraticabile | ~2 h |

Senza il cluster, lo studio di scaling di §5 non sarebbe semplicemente esistito: la seconda metà della campagna (7 run su dataset estesi, più le ablation) è costata in totale meno di una giornata di calcolo.

---

*(Sezioni 5–7 in stesura.)*

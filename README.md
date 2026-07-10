# Clustering-based SSL for Action Discovery

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group**: DataMinds (Lorenzo Comis, Alessandro Sciacca)
- **Project**: Track 12 - Clustering-based SSL for Action Discovery

## 📝 Project Description
Unsupervised discovery of recurring actions in video: a DeepCluster-style **iterative K-Means** loop alternates clustering and pseudo-label fine-tuning of two backbones (ResNet18 and VideoMAE) on a 10-class subset of Kinetics-400, using ground-truth labels **only for evaluation** (purity, NMI, ARI). The project includes a dataset scaling study (398 → 5,802 videos, harvested from the official Kinetics archives) and a fully deterministic, config-driven pipeline: every number in the report can be regenerated from a YAML file and a seed.

> 📖 **Official Report**: For all theoretical details, performance analysis, the architecture used, and group contributions, please refer to our formal paper: **[REPORT.md](docs/REPORT.md)**.

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:**

```bash
git clone https://github.com/Schick01/Clustering-based-SSL-for-Action-Discovery.git
cd Clustering-based-SSL-for-Action-Discovery
conda env create -f environment.yml
conda activate dl-project
```

The environment pins `transformers==4.57.1`: newer 5.x releases silently drop VideoMAE's pretrained attention biases (see the report, chapter 4, and `tests/verify_videomae_weights.py`).

**Dataset:**
Videos live in `data/<class name>/*.mp4` (10 Kinetics-400 classes). The full dataset can be rebuilt from the official Kinetics-400 archives with the streaming downloader, which keeps only the videos of the 10 target classes and never uses more than ~2 GB of temporary disk space:

```bash
# first extension (val + test splits, ~1,900 videos total)
python -m src.datasets.download_kinetics_subset --splits val test

# second extension (train split, stops at 500 videos per class, ~5,800 total)
python -m src.datasets.download_kinetics_subset --splits train --target-per-class 500
```

The per-video frame cache (`data/frame_cache/`) is built automatically on first use.

### 2. Training

**Baseline (feature extraction + single K-Means):**

```bash
python main.py
```

**Iterative clustering:**

```bash
python -m src.training.run_iterative --config experiments/configs/iterative_resnet_gentle.yaml
python -m src.training.run_iterative --config experiments/configs/iterative_videomae.yaml
```

Optional flags: `--device cuda` (default is the `device` set in the config) and `--run-name NAME` (to keep artifacts of multiple runs separate). Available configurations:

| Config | Description |
| :--- | :--- |
| `iterative_resnet_gentle.yaml` | ResNet18, gentle regime (recommended) |
| `iterative_resnet.yaml` | ResNet18, aggressive regime (kept as counterexample) |
| `iterative_resnet_gentle_lr3e6.yaml` | ResNet18, learning rate scaled for the extended dataset |
| `iterative_videomae.yaml` | VideoMAE, standard single-epoch regime |
| `iterative_videomae_lr3e6.yaml` | VideoMAE, learning rate scaled for the extended dataset |
| `iterative_videomae_k50.yaml` | Ablation: over-clustering (K=50) |
| `iterative_videomae_blocks4.yaml` | Ablation: more capacity (4 unlocked blocks) |

### 3. Evaluation

Evaluation is built into every run: purity, NMI and ARI are computed at each iteration (ground-truth labels are used only at this stage) and stored, together with cluster assignments and the config used, in `experiments/logs/<run-name>/history.json`. The baseline in `main.py` prints the same three metrics. All report figures are regenerated from the run histories with:

```bash
python -m src.utils.plot_results
```

**Results on the extended dataset** (5,802 videos, dataset-scaled learning rate; full scaling study and analysis in the [report](docs/REPORT.md)):

| Backbone | Method | Purity | NMI | ARI |
| :--- | :--- | :---: | :---: | :---: |
| ResNet18 | K-Means baseline (+ L2 norm) | 0.596 | 0.521 | 0.385 |
| ResNet18 | **Iterative clustering** | **0.607** | **0.532** | **0.422** |
| VideoMAE | K-Means baseline (+ L2 norm) | 0.311 | 0.221 | 0.119 |
| VideoMAE | **Iterative clustering** | **0.319** | **0.241** | **0.135** |

### 4. Verification Scripts

Three standalone checks guard the correctness of the pipeline (plain Python, no extra dependencies; each exits non-zero on failure):

```bash
python -m tests.verify_frame_cache        # cached frames produce identical features to direct video decoding
python -m tests.verify_videomae_weights   # VideoMAE pretrained attention biases match the official checkpoint
python -m tests.verify_determinism        # two identical runs produce bit-identical histories, assignments and weights
```

`verify_determinism` accepts `--backbone {resnet,videomae}` and `--device {cpu,cuda}` and should be re-run once on any new hardware before running experiments.

---

*For the declaration of individual tasks and the use of AI, refer to `docs/REPORT.md`.*

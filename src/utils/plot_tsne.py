"""
Visualizzazioni t-SNE degli embedding, per il report e le slide.

Genera due figure in ``figures/``:

1. ``tsne_baselines.png``: gli embedding baseline di ResNet18 e VideoMAE
   (398 video, feature salvate in ``data/``) proiettati in 2D, colorati
   con la classe reale. Mostra visivamente il divario tra i due
   backbone: isole nette contro nuvola indistinta.
2. ``tsne_resnet_before_after.png``: gli embedding ResNet prima e dopo
   il fine-tuning iterativo (regime gentile, checkpoint del run a 398
   video). Mostra l'effetto del loop sullo spazio delle feature.

Le etichette reali sono usate SOLO per colorare i punti: è
visualizzazione diagnostica, non training. Il t-SNE è qualitativo
(distanze tra isole e dimensioni non hanno significato metrico) e viene
eseguito con seed fisso su feature L2-normalizzate, coerentemente con
la geometria usata dal clustering.

I 398 video originali vengono individuati tra i 5.802 correnti tramite
la data dei file (le estensioni del dataset sono successive) e
l'identificazione è verificata in due modi: distribuzione per classe
esatta e confronto delle feature ricalcolate con quelle salvate.

Uso (dalla radice del repository):
    python -m src.utils.plot_tsne
"""

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.datasets.frame_cache import load_or_build_frames
from src.datasets.video_transforms import clip_to_float, normalize_for_resnet
from src.models.resnet_baseline import ResNetFeatureExtractor
from src.training.cluster_assignment import l2_normalize
from src.utils.seed import set_seed

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
OUT_DIR = "figures"
CHECKPOINT = os.path.join("experiments", "checkpoints", "iterative_resnet_gentle.pt")
# I video originali sono stati scaricati prima di questa data; le
# estensioni del dataset (val/test/train di Kinetics) sono successive
ORIGINAL_CUTOFF = datetime(2026, 7, 5).timestamp()
# Distribuzione per classe attesa dei 398 originali (ordine alfabetico)
EXPECTED_COUNTS = [45, 38, 35, 35, 34, 43, 44, 42, 37, 45]


def original_video_paths(data_root: str = "data") -> tuple[list[str], list[str]]:
    """
    Individua i 398 video originali tra quelli correnti, preservando
    l'ordine di enumerazione del dataset (stesso ordine delle feature
    salvate).

    Returns:
        Coppia (percorsi, etichette), nell'ordine originale.

    Raises:
        RuntimeError: se la distribuzione per classe non corrisponde a
            quella attesa dei 398 video originali.
    """
    paths: list[str] = []
    labels: list[str] = []
    for class_dir in os.listdir(data_root):
        full_dir = os.path.join(data_root, class_dir)
        if not os.path.isdir(full_dir) or class_dir in ("frame_cache",) or class_dir.startswith("_"):
            continue
        for video in os.listdir(full_dir):
            path = os.path.join(full_dir, video)
            if video.endswith("mp4") and os.path.getmtime(path) < ORIGINAL_CUTOFF:
                paths.append(path)
                labels.append(class_dir)

    counts = [labels.count(c) for c in sorted(set(labels))]
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Sottoinsieme originale non riconosciuto: {counts} != {EXPECTED_COUNTS}")
    return paths, labels


def extract_resnet_features(model: ResNetFeatureExtractor, paths: list[str]) -> np.ndarray:
    """
    Estrae gli embedding ResNet dei video indicati, dalla cache frame.

    Args:
        model: estrattore (già nei pesi desiderati), verrà messo in eval.
        paths: percorsi dei video.

    Returns:
        Matrice (N, 512) delle feature.
    """
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(paths), 8):
            batch = [
                normalize_for_resnet(clip_to_float(load_or_build_frames(p)))
                for p in paths[start : start + 8]
            ]
            chunks.append(model(torch.stack(batch)))
    return torch.cat(chunks).numpy()


def tsne_2d(features: np.ndarray) -> np.ndarray:
    """Proietta le feature (L2-normalizzate) in 2D con seed fisso."""
    return TSNE(
        n_components=2, random_state=42, init="pca", perplexity=30, learning_rate="auto"
    ).fit_transform(l2_normalize(features))


def scatter_panel(ax: plt.Axes, points: np.ndarray, labels: list[str], title: str) -> None:
    """Disegna un pannello t-SNE colorato per classe reale."""
    class_names = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for index, name in enumerate(class_names):
        mask = np.array([lab == name for lab in labels])
        ax.scatter(points[mask, 0], points[mask, 1], s=16, alpha=0.85,
                   color=cmap(index), label=name, linewidths=0)
    ax.set_title(title, fontsize=13, color=INK)
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#e1e0d9")


def make_figure(panels: list[tuple[np.ndarray, list[str], str]], suptitle: str, name: str) -> None:
    """Compone una figura a due pannelli con legenda comune in basso."""
    fig = plt.figure(figsize=(12, 5.6), facecolor=SURFACE)
    fig.suptitle(suptitle, fontsize=15, color=INK, fontweight="bold")

    handles = None
    for i, (points, labels, title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1)
        scatter_panel(ax, points, labels, title)
        if handles is None:
            handles, hlabels = ax.get_legend_handles_labels()

    fig.legend(handles, hlabels, frameon=False, fontsize=10, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.04), markerscale=1.8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"salvata {path}")


def main() -> None:
    """Genera le due figure t-SNE."""
    set_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Figura 1: baseline dei due backbone (feature salvate) ---
    resnet_feats = torch.load("data/features_resnet18.pt", weights_only=True).numpy()
    resnet_labels = torch.load("data/labels_resnet18.pt", weights_only=False)
    videomae_feats = torch.load("data/features_videomae.pt", weights_only=True).numpy()
    videomae_labels = torch.load("data/labels_videomae.pt", weights_only=False)

    print("t-SNE baseline ResNet...")
    resnet_2d = tsne_2d(resnet_feats)
    print("t-SNE baseline VideoMAE...")
    videomae_2d = tsne_2d(videomae_feats)
    make_figure(
        [(resnet_2d, resnet_labels, "ResNet18 (supervisionata su ImageNet)"),
         (videomae_2d, videomae_labels, "VideoMAE (masked autoencoding)")],
        "Gli embedding visti dal t-SNE: la struttura di partenza dei due backbone (398 video)",
        "tsne_baselines.png",
    )

    # --- Figura 2: ResNet prima e dopo il loop iterativo ---
    paths, labels = original_video_paths()
    print(f"sottoinsieme originale: {len(paths)} video")

    # Verifica di allineamento: le feature del backbone NON fine-tunato
    # sui primi video del sottoinsieme devono coincidere con quelle
    # salvate, a conferma che il sottoinsieme e l'ordine sono giusti
    pristine = ResNetFeatureExtractor()
    sample = extract_resnet_features(pristine, paths[:16])
    diff = np.abs(sample - resnet_feats[:16]).max()
    if diff > 1e-4:
        raise RuntimeError(f"Sottoinsieme non allineato alle feature salvate (diff {diff:.2e})")
    print(f"allineamento verificato (diff {diff:.2e})")

    finetuned = ResNetFeatureExtractor()
    finetuned.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    print("estrazione feature col backbone fine-tunato...")
    after_feats = extract_resnet_features(finetuned, paths)

    print("t-SNE prima/dopo...")
    after_2d = tsne_2d(after_feats)
    make_figure(
        [(resnet_2d, resnet_labels, "Prima: backbone pre-addestrato (baseline)"),
         (after_2d, labels, "Dopo: 10 iterazioni di fine-tuning gentile")],
        "L'effetto del loop iterativo sullo spazio delle feature (ResNet18, 398 video)",
        "tsne_resnet_before_after.png",
    )


if __name__ == "__main__":
    main()

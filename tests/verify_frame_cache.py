"""
Verifica di fedeltà della cache dei frame.

Confronta, su un campione di video distribuito su tutto il dataset, i
tensori di input e le feature estratte lungo i due percorsi:

1. decodifica diretta del video (``VideoKineticsDataset``, come nella
   pipeline di estrazione originale);
2. lettura dalla cache uint8 + trasformazioni tensoriali
   (``CachedVideoDataset``).

I due percorsi devono produrre risultati identici a meno di differenze
numeriche trascurabili, per entrambi i backbone. In caso contrario la
cache ha un bug e lo script esce con codice diverso da zero.

Uso (dalla radice del repository):
    python -m tests.verify_frame_cache [--num-videos 8]
"""

import argparse
import sys

import torch

from src.datasets.cached_video_dataset import CachedVideoDataset
from src.datasets.video_dataset import VideoKineticsDataset
from src.models.resnet_baseline import ResNetFeatureExtractor
from src.models.videomae_extractor import VideoMAEFeatureExtractor
from src.utils.seed import set_seed

# Tolleranza sulle differenze assolute massime: i due percorsi eseguono
# le stesse operazioni ma con librerie diverse (numpy/PIL vs torch),
# quindi si ammettono discrepanze all'ultimo bit del float32
TOLERANCE = 1e-5


def sample_indices(dataset_size: int, num_videos: int) -> list[int]:
    """
    Sceglie indici equispaziati sull'intero dataset.

    L'equispaziatura è deterministica e, dato che i video sono elencati
    per classe, garantisce che il campione copra più classi.

    Args:
        dataset_size: numero totale di video.
        num_videos: numero di video da campionare.

    Returns:
        Lista di indici crescenti.
    """
    step = max(1, dataset_size // num_videos)
    return list(range(0, dataset_size, step))[:num_videos]


def verify_backbone(
    backbone_name: str,
    direct_dataset: torch.utils.data.Dataset,
    cached_dataset: CachedVideoDataset,
    model: torch.nn.Module,
    indices: list[int],
    num_feature_checks: int,
) -> bool:
    """
    Confronta input e feature dei due percorsi per un backbone.

    Il confronto sugli input copre tutti gli indici; quello sulle
    feature (che richiede un forward del modello, costoso su CPU per
    VideoMAE) si limita ai primi ``num_feature_checks`` indici.

    Args:
        backbone_name: nome del backbone, solo per i messaggi di log.
        direct_dataset: dataset che decodifica i video con OpenCV.
        cached_dataset: dataset che legge dalla cache dei frame.
        model: estrattore di feature corrispondente.
        indices: indici dei video da confrontare.
        num_feature_checks: quanti indici sottoporre anche al confronto
            delle feature.

    Returns:
        True se tutte le differenze rientrano nella tolleranza.
    """
    all_ok = True
    model.eval()

    for position, idx in enumerate(indices):
        direct_clip, direct_label = direct_dataset[idx]
        cached_clip, cached_label, cached_idx = cached_dataset[idx]

        if direct_label != cached_label or cached_idx != idx:
            print(f"[{backbone_name}] video {idx}: DISALLINEAMENTO etichetta/indice")
            all_ok = False
            continue

        input_diff = (direct_clip - cached_clip).abs().max().item()
        input_ok = input_diff <= TOLERANCE

        feature_diff = None
        feature_ok = True
        if position < num_feature_checks:
            with torch.no_grad():
                direct_features = model(direct_clip.unsqueeze(0))
                cached_features = model(cached_clip.unsqueeze(0))
            feature_diff = (direct_features - cached_features).abs().max().item()
            feature_ok = feature_diff <= TOLERANCE

        status = "OK" if (input_ok and feature_ok) else "FAIL"
        feature_msg = f", diff feature = {feature_diff:.2e}" if feature_diff is not None else ""
        print(
            f"[{backbone_name}] video {idx:3d} ({direct_label}): "
            f"diff input = {input_diff:.2e}{feature_msg} [{status}]"
        )

        all_ok = all_ok and input_ok and feature_ok

    return all_ok


def main() -> int:
    """Esegue la verifica per entrambi i backbone e riassume l'esito."""
    parser = argparse.ArgumentParser(description="Verifica di fedeltà della cache dei frame")
    parser.add_argument("--num-videos", type=int, default=8, help="numero di video da campionare")
    args = parser.parse_args()

    set_seed(42)

    print("=== Verifica cache: percorso ResNet ===")
    direct_resnet = VideoKineticsDataset()
    cached_resnet = CachedVideoDataset(backbone="resnet", train=False)
    indices = sample_indices(len(cached_resnet), args.num_videos)
    resnet_ok = verify_backbone(
        "resnet",
        direct_resnet,
        cached_resnet,
        ResNetFeatureExtractor(),
        indices,
        num_feature_checks=len(indices),
    )

    print("\n=== Verifica cache: percorso VideoMAE ===")
    from transformers import VideoMAEImageProcessor

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    direct_videomae = VideoKineticsDataset(processor=processor)
    cached_videomae = CachedVideoDataset(backbone="videomae", train=False)
    # Il forward di VideoMAE su CPU è costoso: il confronto delle
    # feature si limita a metà del campione
    videomae_ok = verify_backbone(
        "videomae",
        direct_videomae,
        cached_videomae,
        VideoMAEFeatureExtractor(),
        indices,
        num_feature_checks=max(1, len(indices) // 2),
    )

    print()
    if resnet_ok and videomae_ok:
        print(f"VERIFICA SUPERATA: cache fedele per entrambi i backbone (tolleranza {TOLERANCE:.0e})")
        return 0
    print("VERIFICA FALLITA: la cache NON riproduce il percorso originale, correggere prima di proseguire")
    return 1


if __name__ == "__main__":
    sys.exit(main())

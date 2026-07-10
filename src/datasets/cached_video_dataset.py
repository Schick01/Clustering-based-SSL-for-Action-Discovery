"""
Dataset dei video Kinetics basato sulla cache di frame uint8.

A differenza di ``VideoKineticsDataset``, che decodifica i video con
OpenCV a ogni accesso, questo dataset legge i frame dalla cache su disco
(costruendola in modo lazy al primo accesso) e applica le trasformazioni
tensoriali al volo. Supporta due modalità:

- ``train=True``: applica l'augmentation spaziale (per il fine-tuning
  sulle pseudo-label);
- ``train=False``: nessuna augmentation, output deterministico (per
  l'estrazione delle feature).

Oltre a clip ed etichetta reale, restituisce l'indice del campione,
necessario per allineare le pseudo-label ai video durante il training
iterativo. L'etichetta reale non va mai usata in training: è esposta
solo per la valutazione a posteriori.
"""

import torch
from torch.utils.data import Dataset

from src.datasets.frame_cache import DEFAULT_CACHE_ROOT, load_or_build_frames
from src.datasets.video_dataset import get_dataset
from src.datasets.video_transforms import (
    augment_clip,
    clip_to_float,
    normalize_for_resnet,
    normalize_for_videomae,
)


class CachedVideoDataset(Dataset):
    """
    Dataset che serve i clip video dalla cache dei frame decodificati.

    Args:
        backbone: "resnet" o "videomae"; determina layout e
            normalizzazione del clip restituito.
        train: se True applica l'augmentation spaziale al clip.
        path: cartella radice del dataset.
        cache_root: cartella radice della cache dei frame.
        num_frames: numero di frame campionati per video.
    """

    def __init__(
        self,
        backbone: str,
        train: bool = False,
        path: str = "data",
        cache_root: str = DEFAULT_CACHE_ROOT,
        num_frames: int = 16,
    ) -> None:
        if backbone not in ("resnet", "videomae"):
            raise ValueError(f"Backbone non supportato: '{backbone}' (attesi: 'resnet', 'videomae')")

        # Stessa enumerazione dei video del dataset originale, per
        # mantenere identico l'ordinamento dei campioni
        self.video_items = list(get_dataset(path).items())
        self.backbone = backbone
        self.train = train
        self.path = path
        self.cache_root = cache_root
        self.num_frames = num_frames

        if backbone == "videomae":
            # Import locale: evita di caricare transformers quando si usa
            # il percorso ResNet. Mean/std vengono letti dal processor
            # ufficiale per replicarne esattamente la normalizzazione.
            from transformers import VideoMAEImageProcessor

            processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            self.image_mean: list[float] = list(processor.image_mean)
            self.image_std: list[float] = list(processor.image_std)

    def __len__(self) -> int:
        return len(self.video_items)

    def labels(self) -> list[str]:
        """
        Restituisce le etichette reali di tutti i video, nell'ordine del
        dataset. Da usare esclusivamente in fase di valutazione.
        """
        return [action for _, action in self.video_items]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        """
        Restituisce il clip trasformato, l'etichetta reale e l'indice.

        Args:
            idx: indice del video nel dataset.

        Returns:
            Tupla (clip, etichetta reale, indice). Il layout del clip
            dipende dal backbone: (C, T, H, W) per ResNet,
            (T, C, H, W) per VideoMAE.
        """
        video_path, action = self.video_items[idx]

        clip = load_or_build_frames(
            video_path,
            num_frames=self.num_frames,
            dataset_root=self.path,
            cache_root=self.cache_root,
        )
        clip = clip_to_float(clip)

        if self.train:
            clip = augment_clip(clip)

        if self.backbone == "resnet":
            clip = normalize_for_resnet(clip)
        else:
            clip = normalize_for_videomae(clip, self.image_mean, self.image_std)

        return clip, action, idx

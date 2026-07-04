"""
Trasformazioni tensoriali per i clip video letti dalla cache dei frame.

I clip arrivano dalla cache come tensori uint8 (T, H, W, C) in RGB e
vengono convertiti, eventualmente aumentati (solo in training) e
normalizzati nel formato atteso dal backbone di destinazione. I due
percorsi di normalizzazione replicano esattamente quelli della pipeline
di estrazione originale (``VideoKineticsDataset``):

- ResNet: scala in [0, 1] e layout (C, T, H, W), senza normalizzazione
  mean/std (coerente con il percorso senza processor del dataset
  originale);
- VideoMAE: scala in [0, 1], normalizzazione con mean/std del processor
  HuggingFace e layout (T, C, H, W).
"""

from typing import Sequence

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def clip_to_float(clip: torch.Tensor) -> torch.Tensor:
    """
    Converte un clip uint8 (T, H, W, C) in float32 (T, C, H, W) in [0, 1].

    Args:
        clip: tensore uint8 di forma (T, H, W, C).

    Returns:
        Tensore float32 di forma (T, C, H, W) con valori in [0, 1].
    """
    return clip.permute(0, 3, 1, 2).float() / 255.0


def augment_clip(
    clip: torch.Tensor,
    flip_prob: float = 0.5,
    crop_scale: tuple[float, float] = (0.6, 1.0),
) -> torch.Tensor:
    """
    Applica l'augmentation spaziale leggera a un clip (T, C, H, W).

    Le trasformazioni (flip orizzontale e random resized crop) vengono
    campionate una sola volta per clip e applicate in modo identico a
    tutti i frame, per preservare la coerenza temporale del video.
    Da usare solo in training, mai in fase di estrazione feature.

    Args:
        clip: tensore float di forma (T, C, H, W).
        flip_prob: probabilità del flip orizzontale.
        crop_scale: intervallo della frazione di area del crop.

    Returns:
        Clip aumentato, stessa forma dell'input.
    """
    if torch.rand(1).item() < flip_prob:
        clip = clip.flip(-1)

    # Un solo crop campionato per clip: F.resized_crop lo applica a
    # tutti i frame insieme trattando T come dimensione batch
    top, left, height, width = transforms.RandomResizedCrop.get_params(
        clip, scale=list(crop_scale), ratio=[3.0 / 4.0, 4.0 / 3.0]
    )
    clip = F.resized_crop(
        clip,
        top,
        left,
        height,
        width,
        size=[clip.shape[-2], clip.shape[-1]],
        interpolation=InterpolationMode.BILINEAR,
    )

    return clip


def normalize_for_resnet(clip: torch.Tensor) -> torch.Tensor:
    """
    Porta un clip (T, C, H, W) in [0, 1] nel formato atteso dalla ResNet.

    Nessuna normalizzazione mean/std: il percorso ResNet della pipeline
    originale usa i valori grezzi in [0, 1], e la cache deve produrre
    esattamente gli stessi tensori.

    Args:
        clip: tensore float di forma (T, C, H, W) in [0, 1].

    Returns:
        Tensore float di forma (C, T, H, W).
    """
    return clip.permute(1, 0, 2, 3)


def normalize_for_videomae(
    clip: torch.Tensor,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> torch.Tensor:
    """
    Normalizza un clip (T, C, H, W) in [0, 1] come il processor VideoMAE.

    Replica la normalizzazione di ``VideoMAEImageProcessor`` (rescale in
    [0, 1] già applicato a monte, poi sottrazione della media e divisione
    per la deviazione standard canale per canale). I valori di mean/std
    vanno letti dal processor stesso, non hardcodati.

    Args:
        clip: tensore float di forma (T, C, H, W) in [0, 1].
        image_mean: media per canale usata dal processor.
        image_std: deviazione standard per canale usata dal processor.

    Returns:
        Tensore float di forma (T, C, H, W) normalizzato.
    """
    mean = torch.tensor(image_mean, dtype=clip.dtype).view(1, -1, 1, 1)
    std = torch.tensor(image_std, dtype=clip.dtype).view(1, -1, 1, 1)
    return (clip - mean) / std

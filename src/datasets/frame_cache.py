"""
Cache su disco dei frame video decodificati.

La decodifica dei video con OpenCV è il collo di bottiglia della
pipeline iterativa, che deve ripassare l'intero dataset a ogni epoca di
ogni iterazione. Qui i frame campionati da ciascun video vengono
decodificati una sola volta e salvati su disco come tensori uint8
(~2,4 MB a video); le letture successive caricano direttamente i
tensori, saltando OpenCV.

La decodifica replica esattamente quella di
``src.datasets.video_dataset.VideoKineticsDataset``: stessi indici di
campionamento, stesso resize a 224x224, stessa conversione BGR->RGB e
stesso frame nero in caso di lettura fallita. La fedeltà dei due
percorsi è verificata da ``tests/verify_frame_cache.py``.
"""

import os

import cv2
import numpy as np
import torch

FRAME_SIZE = 224
DEFAULT_CACHE_ROOT = os.path.join("data", "frame_cache")


def decode_video_frames(video_path: str, num_frames: int = 16) -> torch.Tensor:
    """
    Decodifica un video campionando ``num_frames`` frame uniformi.

    Args:
        video_path: percorso del file video.
        num_frames: numero di frame da campionare uniformemente.

    Returns:
        Tensore uint8 di forma (num_frames, 224, 224, 3), canali RGB.
    """
    capture = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    for frame_idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # Frame nero in caso di lettura fallita, come nel dataset originale
            frames.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))

    capture.release()

    return torch.from_numpy(np.stack(frames))


def cache_path_for(
    video_path: str,
    dataset_root: str = "data",
    cache_root: str = DEFAULT_CACHE_ROOT,
) -> str:
    """
    Restituisce il percorso del file di cache corrispondente a un video.

    La cache rispecchia la struttura del dataset: il video
    ``data/<classe>/<nome>.mp4`` viene cachato in
    ``<cache_root>/<classe>/<nome>.pt``.

    Args:
        video_path: percorso del file video.
        dataset_root: cartella radice del dataset.
        cache_root: cartella radice della cache.

    Returns:
        Percorso del file ``.pt`` di cache.
    """
    relative_path = os.path.relpath(video_path, dataset_root)
    return os.path.join(cache_root, os.path.splitext(relative_path)[0] + ".pt")


def load_or_build_frames(
    video_path: str,
    num_frames: int = 16,
    dataset_root: str = "data",
    cache_root: str = DEFAULT_CACHE_ROOT,
) -> torch.Tensor:
    """
    Restituisce i frame di un video dalla cache, costruendola se assente.

    Se il file di cache non esiste (o contiene un numero di frame
    diverso da quello richiesto) il video viene decodificato e il
    risultato salvato per gli accessi successivi.

    Args:
        video_path: percorso del file video.
        num_frames: numero di frame attesi per video.
        dataset_root: cartella radice del dataset.
        cache_root: cartella radice della cache.

    Returns:
        Tensore uint8 di forma (num_frames, 224, 224, 3), canali RGB.
    """
    cache_file = cache_path_for(video_path, dataset_root, cache_root)

    if os.path.exists(cache_file):
        frames = torch.load(cache_file, weights_only=True)
        if frames.shape[0] == num_frames:
            return frames
        # Cache costruita con un numero di frame diverso: si rigenera

    frames = decode_video_frames(video_path, num_frames)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save(frames, cache_file)
    return frames

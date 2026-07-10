"""
Fine-tuning del backbone sulle pseudo-label di una singola iterazione.

È la fase "supervisionata dai cluster" del loop iterativo: le
pseudo-label sono le assegnazioni K-Means correnti e non contengono
alcuna informazione sulle etichette reali (che il dataset restituisce
ma vengono esplicitamente scartate).

La cross-entropy è pesata con l'inverso della dimensione dei cluster
per contrastare il collasso: senza pesatura i cluster grandi
dominerebbero il gradiente e verrebbero rinforzati a ogni giro, fino a
degenerare in pochi mega-cluster.
"""

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.finetuning import freeze_batchnorm_stats
from src.utils.seed import make_generator, seed_worker


def compute_cluster_weights(
    pseudo_labels: np.ndarray,
    num_clusters: int,
) -> torch.Tensor:
    """
    Calcola i pesi di classe inversamente proporzionali alla dimensione
    dei cluster: ``w_k = N / (K * n_k)``.

    Con cluster perfettamente bilanciati tutti i pesi valgono 1; i
    cluster piccoli pesano di più, quelli grandi di meno. Ai cluster
    vuoti viene assegnato peso 0 (nessun campione li usa comunque).

    Args:
        pseudo_labels: assegnazioni ai cluster, una per campione.
        num_clusters: numero totale di cluster K.

    Returns:
        Tensore float32 di forma (num_clusters,) con i pesi di classe.
    """
    counts = np.bincount(pseudo_labels, minlength=num_clusters)
    total = len(pseudo_labels)

    weights = np.where(
        counts > 0,
        total / (num_clusters * np.maximum(counts, 1)),
        0.0,
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_on_pseudo_labels(
    backbone: nn.Module,
    head: nn.Module,
    train_dataset: Dataset,
    pseudo_labels: np.ndarray,
    trainable_params: Sequence[nn.Parameter],
    num_clusters: int,
    epochs: int = 2,
    batch_size: int = 4,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> list[float]:
    """
    Esegue il fine-tuning di backbone e testa sulle pseudo-label.

    Backbone e testa vengono aggiornati in place; l'ottimizzatore usa
    due learning rate distinti (basso per il backbone pre-addestrato,
    alto per la testa appena inizializzata).

    Args:
        backbone: estrattore di feature, già preparato con lo sblocco
            selettivo dei parametri.
        head: testa lineare di classificazione.
        train_dataset: dataset in modalità training (con augmentation),
            che restituisce (clip, etichetta reale, indice).
        pseudo_labels: assegnazioni K-Means correnti, una per campione,
            allineate agli indici del dataset.
        trainable_params: parametri allenabili del backbone.
        num_clusters: numero totale di cluster K.
        epochs: numero di epoche di training.
        batch_size: dimensione del batch.
        lr_backbone: learning rate per i parametri del backbone.
        lr_head: learning rate per la testa.
        weight_decay: weight decay di AdamW.
        device: dispositivo di calcolo ("cpu" o "cuda").
        seed: seed per lo shuffling del DataLoader (variarlo tra le
            iterazioni del loop evita ordini di visita identici).

    Returns:
        Lista della loss media per epoca.
    """
    backbone.to(device)
    head.to(device)

    targets = torch.from_numpy(np.asarray(pseudo_labels)).long()

    class_weights = compute_cluster_weights(pseudo_labels, num_clusters).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(trainable_params), "lr": lr_backbone},
            {"params": list(head.parameters()), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
    )

    epoch_losses = []
    for _ in range(epochs):
        backbone.train()
        head.train()
        # Le statistiche BatchNorm restano congelate anche in training
        freeze_batchnorm_stats(backbone)

        total_loss = 0.0
        num_batches = 0

        # L'etichetta reale restituita dal dataset viene scartata:
        # il training vede esclusivamente le pseudo-label
        for clips, _, indices in loader:
            clips = clips.to(device)
            batch_targets = targets[indices].to(device)

            optimizer.zero_grad()
            logits = head(backbone(clips))
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_losses.append(total_loss / num_batches)

    return epoch_losses

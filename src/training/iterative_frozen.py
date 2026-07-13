"""
Loop di clustering iterativo a backbone congelato (metodo alternativo).

Variante economica del loop completo (``iterative_clustering``): il
backbone non viene mai aggiornato, le sue feature vengono estratte UNA
sola volta e tutto il loop lavora sulla matrice (N, D) in memoria. A
imparare e' un piccolo MLP di proiezione (``ProjectionHead``), allenato
sulle pseudo-label insieme a un classificatore lineare usa-e-getta; il
clustering delle iterazioni successive avviene sull'uscita della
proiezione.

Differenze dichiarate rispetto al metodo completo:

- l'iterazione 0 (K-Means sulle feature grezze) coincide con la stessa
  baseline del metodo completo, per costruzione;
- dall'iterazione 1 il clustering avviene nello spazio di proiezione;
- nessuna augmentation: le feature sono fisse, l'augmentation agisce
  sui pixel; la regolarizzazione e' affidata al weight decay.

Il protocollo e' identico al loop completo: pseudo-label dal K-Means,
cross-entropy pesata anti-collasso, stop su stabilita' NMI tra
assegnazioni consecutive, etichette reali SOLO nella valutazione a
posteriori, risultato = ultima iterazione. Lo storico ``history.json``
usa lo stesso formato, cosi' gli strumenti di analisi esistenti
funzionano invariati.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.cached_video_dataset import CachedVideoDataset
from src.evaluation.metrics import nmi_score
from src.models.finetuning import build_linear_head
from src.models.projection_head import ProjectionHead
from src.models.resnet_baseline import ResNetFeatureExtractor
from src.models.videomae_extractor import VideoMAEFeatureExtractor
from src.training.cluster_assignment import assign_clusters, report_cluster_sizes
from src.training.iterative_clustering import (
    CHECKPOINTS_ROOT,
    LOGS_ROOT,
    evaluate_iteration,
    extract_features,
)
from src.training.pseudo_label_training import compute_cluster_weights
from src.utils.seed import make_generator, seed_worker


def load_or_extract_features(config: dict) -> np.ndarray:
    """
    Restituisce la matrice delle feature del backbone congelato.

    Se ``features_path`` e' indicato ed esiste, le feature vengono
    caricate da disco; altrimenti vengono estratte con il backbone
    pre-addestrato (l'unico passo costoso del metodo) e, se
    ``features_path`` e' indicato, salvate per i run successivi.

    Args:
        config: configurazione del run (chiavi: "backbone", "data_path",
            "device", "extract_batch_size" e opzionalmente
            "features_path").

    Returns:
        Matrice (N, D) delle feature in float32.
    """
    features_path = config.get("features_path")
    if features_path and os.path.exists(features_path):
        print(f"Feature caricate da {features_path}")
        return torch.load(features_path, weights_only=True).numpy()

    if config["backbone"] == "resnet":
        backbone: nn.Module = ResNetFeatureExtractor()
    elif config["backbone"] == "videomae":
        backbone = VideoMAEFeatureExtractor()
    else:
        raise ValueError(f"Backbone non supportato: '{config['backbone']}'")

    dataset = CachedVideoDataset(
        backbone=config["backbone"], train=False, path=config["data_path"]
    )
    loader = DataLoader(dataset, batch_size=config["extract_batch_size"], shuffle=False)
    features = extract_features(backbone, loader, config["device"])

    if features_path:
        features_dir = os.path.dirname(features_path)
        if features_dir:
            os.makedirs(features_dir, exist_ok=True)
        torch.save(torch.from_numpy(features), features_path)
        print(f"Feature salvate in {features_path}")

    return features


def train_projection_on_pseudo_labels(
    projection: ProjectionHead,
    head: nn.Module,
    features: torch.Tensor,
    pseudo_labels: np.ndarray,
    num_clusters: int,
    epochs: int,
    batch_size: int,
    lr_projection: float,
    lr_head: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> list[float]:
    """
    Allena proiezione e classificatore sulle pseudo-label correnti.

    Speculare a ``train_on_pseudo_labels`` del metodo completo, ma
    l'input sono le feature fisse (vettori), non i clip: ogni epoca
    costa una frazione di secondo.

    Args:
        projection: MLP di proiezione (aggiornato in place, persiste
            tra le iterazioni).
        head: classificatore lineare usa-e-getta.
        features: matrice (N, D) delle feature congelate.
        pseudo_labels: assegnazioni K-Means correnti, una per campione.
        num_clusters: numero totale di cluster K.
        epochs: numero di epoche di training.
        batch_size: dimensione del batch (di vettori).
        lr_projection: learning rate della proiezione.
        lr_head: learning rate del classificatore.
        weight_decay: weight decay di AdamW.
        device: dispositivo di calcolo.
        seed: seed dello shuffling (variato per iterazione dal chiamante).

    Returns:
        Lista della loss media per epoca.
    """
    projection.to(device)
    head.to(device)

    targets = torch.from_numpy(np.asarray(pseudo_labels)).long()

    class_weights = compute_cluster_weights(pseudo_labels, num_clusters).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(projection.parameters()), "lr": lr_projection},
            {"params": list(head.parameters()), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
    )

    epoch_losses = []
    for _ in range(epochs):
        projection.train()
        head.train()

        total_loss = 0.0
        num_batches = 0

        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = head(projection(batch_features))
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_losses.append(total_loss / num_batches)

    return epoch_losses


def project_features(
    projection: ProjectionHead,
    features: torch.Tensor,
    device: str,
) -> np.ndarray:
    """
    Proietta l'intera matrice delle feature nello spazio di clustering.

    Args:
        projection: MLP di proiezione (viene messo in eval).
        features: matrice (N, D) delle feature congelate.
        device: dispositivo di calcolo.

    Returns:
        Matrice (N, projection_dim) in float32.
    """
    projection.to(device)
    projection.eval()
    with torch.no_grad():
        return projection(features.to(device)).cpu().numpy()


def _print_iteration_summary(entry: dict) -> None:
    """Stampa il riepilogo di una singola iterazione."""
    stability = f"{entry['stability']:.4f}" if entry["stability"] is not None else "-"
    losses = ", ".join(f"{loss:.4f}" for loss in entry["epoch_losses"]) or "-"
    print(
        f"[iter {entry['iteration']:2d}] purity = {entry['purity']:.4f} | "
        f"NMI = {entry['nmi']:.4f} | ARI = {entry['ari']:.4f} | "
        f"stabilita = {stability} | loss epoche = [{losses}]"
    )


def run_frozen_clustering(config: dict) -> list[dict]:
    """
    Esegue il loop completo del metodo a backbone congelato.

    Artefatti prodotti, nello stesso formato del metodo completo:
    ``experiments/logs/<run_name>/history.json`` (riscritto a ogni
    iterazione) e le assegnazioni per iterazione; il checkpoint finale
    della proiezione va in ``experiments/checkpoints/<run_name>.pt``.

    Args:
        config: configurazione del run (vedi experiments/configs/).

    Returns:
        Storico per iterazione: metriche, stabilita', loss e dimensioni
        dei cluster.
    """
    device: str = config["device"]
    num_clusters: int = config["num_clusters"]
    seed: int = config["seed"]

    run_dir = os.path.join(LOGS_ROOT, config["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    # Etichette reali: entrano esclusivamente in evaluate_iteration.
    # Il dataset viene istanziato solo per l'enumerazione (lazy), non
    # tocca la cache dei frame
    true_labels = CachedVideoDataset(
        backbone=config["backbone"], train=False, path=config["data_path"]
    ).labels()

    features = load_or_extract_features(config)
    if len(true_labels) != features.shape[0]:
        raise RuntimeError(
            f"Feature e dataset disallineati: {features.shape[0]} feature, "
            f"{len(true_labels)} video (features_path stantio?)"
        )
    features_tensor = torch.from_numpy(features).float()

    projection = ProjectionHead(
        feature_dim=features.shape[1], projection_dim=config["projection_dim"]
    )

    def record_iteration(
        iteration: int,
        assignments: np.ndarray,
        stability: float | None,
        epoch_losses: list[float],
        cluster_sizes: np.ndarray,
        history: list[dict],
    ) -> None:
        """Aggiunge l'iterazione allo storico e salva gli artefatti."""
        entry = {
            "iteration": iteration,
            **evaluate_iteration(true_labels, assignments),
            "stability": stability,
            "epoch_losses": epoch_losses,
            "cluster_sizes": [int(size) for size in cluster_sizes],
        }
        history.append(entry)
        _print_iteration_summary(entry)

        np.save(os.path.join(run_dir, f"assignments_iter{iteration:02d}.npy"), assignments)
        # Lo storico viene riscritto a ogni iterazione: in caso di
        # interruzione i risultati parziali restano su disco
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    history: list[dict] = []

    print("Iterazione 0 (baseline: K-Means sulle feature congelate, stessa del metodo completo)")
    assignments = assign_clusters(features, num_clusters, seed=seed)
    cluster_sizes = report_cluster_sizes(
        assignments, num_clusters, min_size=config["min_cluster_size"]
    )
    record_iteration(0, assignments, None, [], cluster_sizes, history)

    for iteration in range(1, config["max_iterations"] + 1):
        print(f"\nIterazione {iteration}")

        pseudo_labels = assignments
        head = build_linear_head(config["projection_dim"], num_clusters)

        epoch_losses = train_projection_on_pseudo_labels(
            projection=projection,
            head=head,
            features=features_tensor,
            pseudo_labels=pseudo_labels,
            num_clusters=num_clusters,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr_projection=config["lr_projection"],
            lr_head=config["lr_head"],
            weight_decay=config["weight_decay"],
            device=device,
            # Seed diverso per iterazione: ordine di visita variato ma
            # deterministico a parita' di configurazione
            seed=seed + iteration,
        )

        projected = project_features(projection, features_tensor, device)
        new_assignments = assign_clusters(projected, num_clusters, seed=seed)
        cluster_sizes = report_cluster_sizes(
            new_assignments, num_clusters, min_size=config["min_cluster_size"]
        )

        # Criterio di stop interno: confronta SOLO assegnazioni
        # consecutive, le etichette reali non partecipano
        stability = nmi_score(assignments, new_assignments)
        assignments = new_assignments

        record_iteration(iteration, assignments, stability, epoch_losses, cluster_sizes, history)

        if stability >= config["stability_threshold"]:
            print(
                f"Pseudo-label stabili (NMI = {stability:.4f} >= "
                f"{config['stability_threshold']}): stop anticipato"
            )
            break

    os.makedirs(CHECKPOINTS_ROOT, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINTS_ROOT, config["run_name"] + ".pt")
    torch.save(projection.state_dict(), checkpoint_path)
    print(f"\nCheckpoint della proiezione salvato in {checkpoint_path}")
    print(f"Storico del run in {os.path.join(run_dir, 'history.json')}")

    return history

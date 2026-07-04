"""
Loop di clustering iterativo in stile DeepCluster.

Schema: le feature correnti vengono clusterizzate con K-Means; le
assegnazioni diventano pseudo-label per un breve fine-tuning del
backbone; le feature vengono ri-estratte con il backbone aggiornato e
ri-clusterizzate, e così via. L'iterazione 0 (K-Means sulle feature del
modello pre-addestrato, prima di qualunque fine-tuning) coincide con la
baseline. L'ipotesi da verificare sperimentalmente è che il loop renda
le feature progressivamente più clusterizzabili, invece di memorizzare
il rumore delle pseudo-label.

Criterio di stop interno: NMI tra le assegnazioni di due iterazioni
consecutive (invariante alla rinumerazione dei cluster del K-Means).
Le etichette reali non entrano mai nel flusso di controllo: le usa solo
``evaluate_iteration`` per il logging delle metriche, dopo che le
assegnazioni sono già state prodotte.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cached_video_dataset import CachedVideoDataset
from src.evaluation.metrics import ari_score, nmi_score, purity_score
from src.models.finetuning import (
    build_linear_head,
    unfreeze_resnet_top,
    unfreeze_videomae_top,
)
from src.models.resnet_baseline import ResNetFeatureExtractor
from src.models.videomae_extractor import VideoMAEFeatureExtractor
from src.training.cluster_assignment import assign_clusters, report_cluster_sizes
from src.training.pseudo_label_training import train_on_pseudo_labels

LOGS_ROOT = os.path.join("experiments", "logs")
CHECKPOINTS_ROOT = os.path.join("experiments", "checkpoints")


def build_backbone(config: dict) -> tuple[nn.Module, list[nn.Parameter]]:
    """
    Costruisce il backbone richiesto e ne sblocca i layer superiori.

    Args:
        config: configurazione del run (chiavi: "backbone" e, per
            VideoMAE, "num_blocks").

    Returns:
        Coppia (backbone, lista dei parametri allenabili).
    """
    if config["backbone"] == "resnet":
        model: nn.Module = ResNetFeatureExtractor()
        trainable_params = unfreeze_resnet_top(model)
    elif config["backbone"] == "videomae":
        model = VideoMAEFeatureExtractor()
        trainable_params = unfreeze_videomae_top(model, num_blocks=config["num_blocks"])
    else:
        raise ValueError(f"Backbone non supportato: '{config['backbone']}'")

    return model, trainable_params


def extract_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    """
    Estrae le feature dell'intero dataset, nell'ordine degli indici.

    Args:
        backbone: estrattore di feature (viene messo in eval).
        loader: DataLoader del dataset in modalità estrazione, senza
            shuffle, così le feature restano allineate agli indici.
        device: dispositivo di calcolo.

    Returns:
        Matrice (N, D) delle feature in float32.
    """
    backbone.to(device)
    backbone.eval()

    chunks = []
    with torch.no_grad():
        for clips, _, _ in tqdm(loader, desc="Estrazione feature"):
            chunks.append(backbone(clips.to(device)).cpu())

    return torch.cat(chunks, dim=0).numpy()


def evaluate_iteration(
    true_labels: list[str],
    assignments: np.ndarray,
) -> dict[str, float]:
    """
    Calcola le metriche di valutazione rispetto alle etichette reali.

    Serve esclusivamente al logging: i valori restituiti non devono mai
    essere usati per decidere quando fermare il loop o quale iterazione
    scegliere come risultato.

    Args:
        true_labels: etichette reali, una per campione.
        assignments: assegnazioni ai cluster, una per campione.

    Returns:
        Dizionario con purity, NMI e ARI.
    """
    return {
        "purity": purity_score(true_labels, assignments),
        "nmi": nmi_score(true_labels, assignments),
        "ari": ari_score(true_labels, assignments),
    }


def _print_iteration_summary(entry: dict) -> None:
    """Stampa il riepilogo di una singola iterazione."""
    stability = f"{entry['stability']:.4f}" if entry["stability"] is not None else "-"
    losses = ", ".join(f"{loss:.4f}" for loss in entry["epoch_losses"]) or "-"
    print(
        f"[iter {entry['iteration']:2d}] purity = {entry['purity']:.4f} | "
        f"NMI = {entry['nmi']:.4f} | ARI = {entry['ari']:.4f} | "
        f"stabilita = {stability} | loss epoche = [{losses}]"
    )


def run_iterative_clustering(config: dict) -> list[dict]:
    """
    Esegue il loop completo di clustering iterativo.

    A ogni iterazione testa di classificazione e ottimizzatore vengono
    ricreati da zero (la testa perché gli ID dei cluster permutano tra
    iterazioni, l'ottimizzatore perché i momenti di AdamW accumulati
    sulle pseudo-label precedenti non hanno senso su quelle nuove;
    l'ottimizzatore nasce dentro ``train_on_pseudo_labels``).

    Artefatti prodotti in ``experiments/logs/<run_name>/``: lo storico
    ``history.json`` (riscritto a ogni iterazione) e le assegnazioni di
    ogni iterazione. Il checkpoint finale del backbone va in
    ``experiments/checkpoints/<run_name>.pt``.

    Args:
        config: configurazione del run (vedi experiments/configs/).

    Returns:
        Storico per iterazione: metriche, stabilità, loss e dimensioni
        dei cluster.
    """
    device: str = config["device"]
    num_clusters: int = config["num_clusters"]
    seed: int = config["seed"]

    run_dir = os.path.join(LOGS_ROOT, config["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    extract_dataset = CachedVideoDataset(
        backbone=config["backbone"], train=False, path=config["data_path"]
    )
    train_dataset = CachedVideoDataset(
        backbone=config["backbone"], train=True, path=config["data_path"]
    )
    extract_loader = DataLoader(
        extract_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Etichette reali: entrano esclusivamente in evaluate_iteration
    true_labels = extract_dataset.labels()

    backbone, trainable_params = build_backbone(config)

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

    print("Iterazione 0 (baseline: modello pre-addestrato, nessun fine-tuning)")
    features = extract_features(backbone, extract_loader, device)
    assignments = assign_clusters(features, num_clusters, seed=seed)
    cluster_sizes = report_cluster_sizes(
        assignments, num_clusters, min_size=config["min_cluster_size"]
    )
    record_iteration(0, assignments, None, [], cluster_sizes, history)

    for iteration in range(1, config["max_iterations"] + 1):
        print(f"\nIterazione {iteration}")

        pseudo_labels = assignments
        head = build_linear_head(features.shape[1], num_clusters)

        epoch_losses = train_on_pseudo_labels(
            backbone=backbone,
            head=head,
            train_dataset=train_dataset,
            pseudo_labels=pseudo_labels,
            trainable_params=trainable_params,
            num_clusters=num_clusters,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr_backbone=config["lr_backbone"],
            lr_head=config["lr_head"],
            weight_decay=config["weight_decay"],
            device=device,
            # Seed diverso per iterazione: ordine di visita variato ma
            # deterministico a parità di configurazione
            seed=seed + iteration,
        )

        features = extract_features(backbone, extract_loader, device)
        new_assignments = assign_clusters(features, num_clusters, seed=seed)
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
    torch.save(backbone.state_dict(), checkpoint_path)
    print(f"\nCheckpoint del backbone salvato in {checkpoint_path}")
    print(f"Storico del run in {os.path.join(run_dir, 'history.json')}")

    return history

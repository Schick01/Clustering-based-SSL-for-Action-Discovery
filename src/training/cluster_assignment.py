"""
Assegnazione dei cluster tramite K-Means su feature L2-normalizzate.

È l'unico punto della pipeline in cui si clusterizza: sia l'iterazione
0 (baseline) sia le iterazioni successive del loop passano da qui, così
i risultati restano confrontabili per costruzione.

La L2-normalization rende il K-Means equivalente a un clustering per
similarità coseno (spherical K-Means), geometria più adatta alle
feature di reti profonde rispetto alla distanza euclidea grezza.
"""

import numpy as np
from sklearn.cluster import KMeans


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """
    Normalizza ogni riga alla norma unitaria.

    Args:
        features: matrice (N, D) delle feature.

    Returns:
        Matrice (N, D) con righe a norma 1 (le righe a norma nulla
        vengono lasciate a zero anziché produrre divisioni per zero).
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def assign_clusters(
    features: np.ndarray,
    num_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Esegue K-Means sulle feature L2-normalizzate.

    Args:
        features: matrice (N, D) delle feature.
        num_clusters: numero di cluster K.
        seed: seed del K-Means, per la riproducibilità.

    Returns:
        Array (N,) di interi con l'assegnazione di ogni campione.
    """
    normalized = l2_normalize(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(normalized)
    return kmeans.labels_


def report_cluster_sizes(
    assignments: np.ndarray,
    num_clusters: int,
    min_size: int = 5,
) -> np.ndarray:
    """
    Stampa la distribuzione delle dimensioni dei cluster e segnala le
    degenerazioni.

    È il "canarino" del collasso: durante una degenerazione le metriche
    possono anche migliorare mentre un cluster divora gli altri, quindi
    le dimensioni vanno osservate insieme alle metriche.

    Args:
        assignments: array (N,) delle assegnazioni ai cluster.
        num_clusters: numero totale di cluster K.
        min_size: sotto questa dimensione un cluster genera un warning.

    Returns:
        Array (num_clusters,) con la dimensione di ogni cluster.
    """
    sizes = np.bincount(assignments, minlength=num_clusters)
    print(f"Dimensioni cluster: {sizes.tolist()} (min {sizes.min()}, max {sizes.max()})")

    for cluster_id, size in enumerate(sizes):
        if size == 0:
            print(f"ATTENZIONE: cluster {cluster_id} vuoto")
        elif size < min_size:
            print(f"ATTENZIONE: cluster {cluster_id} degenerato ({size} campioni < {min_size})")

    return sizes

"""
Metriche di valutazione per il clustering non supervisionato.

Le etichette reali entrano in gioco solo qui: servono come chiave di
valutazione a posteriori e non vengono mai usate durante il training o
il clustering. Tutte le funzioni sono pure: ricevono le etichette reali
e le assegnazioni ai cluster, restituiscono un valore scalare.
"""

from typing import Hashable, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _validate_inputs(
    true_labels: Sequence[Hashable],
    cluster_labels: Sequence[Hashable],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converte le etichette in array numpy e ne verifica la coerenza.

    Args:
        true_labels: etichette reali, una per campione.
        cluster_labels: assegnazioni ai cluster, una per campione.

    Returns:
        Coppia di array numpy (etichette reali, assegnazioni).

    Raises:
        ValueError: se le sequenze sono vuote o di lunghezza diversa.
    """
    true = np.asarray(true_labels)
    clusters = np.asarray(cluster_labels)

    if true.size == 0:
        raise ValueError("Le sequenze di etichette non possono essere vuote.")
    if true.shape != clusters.shape:
        raise ValueError(
            f"Lunghezze diverse: {true.shape[0]} etichette reali, "
            f"{clusters.shape[0]} assegnazioni ai cluster."
        )

    return true, clusters


def purity_score(
    true_labels: Sequence[Hashable],
    cluster_labels: Sequence[Hashable],
) -> float:
    """
    Calcola la purity del clustering.

    Per ogni cluster si individua l'etichetta reale maggioritaria e si
    contano i campioni che vi coincidono; la purity è la frazione di
    campioni così "corretti" sul totale. Coincide con l'accuracy
    majority-vote della baseline. Da leggere insieme a NMI e ARI perché
    cresce banalmente all'aumentare del numero di cluster.

    Args:
        true_labels: etichette reali, una per campione.
        cluster_labels: assegnazioni ai cluster, una per campione.

    Returns:
        Purity in [0, 1].
    """
    true, clusters = _validate_inputs(true_labels, cluster_labels)

    correct = 0
    for cluster_id in np.unique(clusters):
        labels_in_cluster = true[clusters == cluster_id]
        _, counts = np.unique(labels_in_cluster, return_counts=True)
        correct += counts.max()

    return float(correct) / true.size


def nmi_score(
    true_labels: Sequence[Hashable],
    cluster_labels: Sequence[Hashable],
) -> float:
    """
    Calcola la Normalized Mutual Information tra cluster ed etichette reali.

    Misura teorico-informativa in [0, 1], invariante alle permutazioni
    degli identificativi dei cluster e robusta rispetto al numero di
    cluster: adatta a confrontare iterazioni successive del clustering.

    Args:
        true_labels: etichette reali, una per campione.
        cluster_labels: assegnazioni ai cluster, una per campione.

    Returns:
        NMI in [0, 1].
    """
    true, clusters = _validate_inputs(true_labels, cluster_labels)
    return float(normalized_mutual_info_score(true, clusters))


def ari_score(
    true_labels: Sequence[Hashable],
    cluster_labels: Sequence[Hashable],
) -> float:
    """
    Calcola l'Adjusted Rand Index tra cluster ed etichette reali.

    Misura basata sul conteggio delle coppie di campioni concordi,
    corretta per il caso: vale ~0 per un clustering casuale e 1 per un
    clustering perfetto. Penalizza i cluster spezzati, complementare
    alla NMI.

    Args:
        true_labels: etichette reali, una per campione.
        cluster_labels: assegnazioni ai cluster, una per campione.

    Returns:
        ARI in [-1, 1] (~0 per assegnazioni casuali).
    """
    true, clusters = _validate_inputs(true_labels, cluster_labels)
    return float(adjusted_rand_score(true, clusters))

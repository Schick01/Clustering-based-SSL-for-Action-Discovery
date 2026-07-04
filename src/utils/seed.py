"""
Utility per il seeding centralizzato di tutte le sorgenti di casualità.

L'obiettivo è rendere la pipeline interamente riproducibile: chiamare
``set_seed`` come prima istruzione di ogni entry point garantisce che due
esecuzioni dello stesso codice producano risultati identici.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Inizializza con lo stesso seed tutte le sorgenti di casualità usate
    dalla pipeline: ``random``, ``numpy`` e ``torch`` (CPU e CUDA).

    Forza inoltre cuDNN in modalità deterministica, al costo di una
    possibile riduzione di velocità sulle convoluzioni.

    Args:
        seed: valore del seed da applicare a tutte le librerie.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN di default sceglie a runtime gli algoritmi più veloci, che
    # possono variare tra un run e l'altro: qui si forza la scelta
    # deterministica e si disabilita il benchmark.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int = 42) -> torch.Generator:
    """
    Crea un ``torch.Generator`` con seed fissato, da passare al
    ``DataLoader`` per rendere riproducibile lo shuffling dei dati.

    Args:
        seed: seed del generatore.

    Returns:
        Generatore PyTorch inizializzato con il seed indicato.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """
    Inizializza le sorgenti di casualità di un singolo worker del
    ``DataLoader``.

    Da passare come ``worker_init_fn``: ogni worker riceve da PyTorch un
    seed derivato da quello del processo principale, e qui lo si propaga
    anche a ``numpy`` e ``random``.

    Args:
        worker_id: indice del worker (richiesto dall'interfaccia PyTorch).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

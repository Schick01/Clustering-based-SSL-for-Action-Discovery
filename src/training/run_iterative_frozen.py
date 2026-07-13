"""
Entry point CLI del clustering iterativo a backbone congelato.

Uso (dalla radice del repository):
    python -m src.training.run_iterative_frozen --config experiments/configs/frozen_resnet.yaml

Con ``--run-name`` si puo' dare un nome diverso al run (per esperimenti
alternativi) senza modificare il file di configurazione.
"""

import argparse
import os
import shutil

import yaml

from src.training.iterative_clustering import LOGS_ROOT
from src.training.iterative_frozen import run_frozen_clustering
from src.utils.seed import set_seed


def main() -> None:
    """Carica la configurazione, applica il seed e lancia il loop."""
    parser = argparse.ArgumentParser(
        description="Clustering iterativo a backbone congelato (proiezione MLP)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="percorso del file YAML di configurazione",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="nome del run (sovrascrive run_name della configurazione)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="dispositivo di calcolo, es. 'cuda' (sovrascrive device della configurazione)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.device is not None:
        config["device"] = args.device

    # Seeding centralizzato PRIMA di qualunque operazione con casualita'
    # (inizializzazione di proiezione e testa, shuffle, K-Means)
    set_seed(config["seed"])

    # Copia della configurazione nella cartella del run, per tracciabilita'
    run_dir = os.path.join(LOGS_ROOT, config["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))

    run_frozen_clustering(config)


if __name__ == "__main__":
    main()

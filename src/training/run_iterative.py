"""
Entry point CLI del clustering iterativo.

Uso (dalla radice del repository):
    python -m src.training.run_iterative --config experiments/configs/iterative_resnet.yaml

Con ``--run-name`` si può dare un nome diverso al run (per esperimenti
alternativi) senza modificare il file di configurazione.
"""

import argparse
import os
import shutil

import yaml

from src.training.iterative_clustering import LOGS_ROOT, run_iterative_clustering
from src.utils.seed import set_seed


def main() -> None:
    """Carica la configurazione, applica il seed e lancia il loop."""
    parser = argparse.ArgumentParser(description="Clustering iterativo con fine-tuning del backbone")
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
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.run_name is not None:
        config["run_name"] = args.run_name

    # Seeding centralizzato PRIMA di qualunque operazione con casualità
    # (inizializzazione dei modelli, shuffle, augmentation, K-Means)
    set_seed(config["seed"])

    # Copia della configurazione nella cartella del run, per tracciabilità
    run_dir = os.path.join(LOGS_ROOT, config["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))

    run_iterative_clustering(config)


if __name__ == "__main__":
    main()

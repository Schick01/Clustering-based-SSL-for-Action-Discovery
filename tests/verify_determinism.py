"""
Verifica del determinismo della pipeline iterativa.

Esegue due volte, in processi separati e attraverso il vero entry point
CLI, un mini-run del loop di clustering iterativo (sottoinsieme del
dataset, 2 iterazioni, 1 epoca) e confronta gli artefatti prodotti:

1. ``history.json``: metriche, stabilità e loss identiche;
2. assegnazioni ai cluster di ogni iterazione identiche;
3. checkpoint finali del backbone identici parametro per parametro.

Se anche un solo valore differisce, la catena contiene una sorgente di
casualità non seedata (inizializzazione, shuffle, augmentation, K-Means)
e va corretta prima di qualsiasi esperimento. Esce con codice diverso
da zero in caso di differenze.

Uso (dalla radice del repository):
    python -m tests.verify_determinism [--backbone resnet|videomae] [--device cpu|cuda] [--method full|frozen]

Con ``--method frozen`` viene verificato il metodo a backbone congelato
(``run_iterative_frozen``): ogni mini-run estrae le proprie feature in
un percorso separato, così la verifica copre anche il determinismo
dell'estrazione, oltre a quello del loop sulla proiezione.

Il determinismo va riverificato su ogni combinazione
dispositivo/backbone/metodo usata per gli esperimenti: kernel CUDA e
implementazioni di attention possono introdurre sorgenti di
non-determinismo assenti su CPU.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

from src.training.iterative_clustering import CHECKPOINTS_ROOT, LOGS_ROOT

VIDEOS_PER_CLASS = 2
RUN_NAMES = ("determinism_a", "determinism_b")


def build_mini_dataset(source_root: str, target_root: str) -> int:
    """
    Copia i primi ``VIDEOS_PER_CLASS`` video di ogni classe in una
    cartella temporanea, preservando la struttura ``<classe>/<video>``
    (così la cache dei frame esistente viene riutilizzata).

    Args:
        source_root: cartella radice del dataset completo.
        target_root: cartella di destinazione del mini-dataset.

    Returns:
        Numero di video copiati.
    """
    num_copied = 0
    for class_name in sorted(os.listdir(source_root)):
        class_dir = os.path.join(source_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        videos = sorted(v for v in os.listdir(class_dir) if v.endswith("mp4"))
        os.makedirs(os.path.join(target_root, class_name), exist_ok=True)
        for video in videos[:VIDEOS_PER_CLASS]:
            shutil.copy2(
                os.path.join(class_dir, video),
                os.path.join(target_root, class_name, video),
            )
            num_copied += 1
    return num_copied


def write_check_config(
    path: str,
    data_path: str,
    backbone: str,
    device: str,
    method: str = "full",
    features_path: str | None = None,
) -> None:
    """
    Scrive la configurazione del mini-run di verifica.

    La soglia di stabilità è volutamente irraggiungibile (> 1) così
    entrambi i run eseguono sempre tutte le iterazioni previste.

    Args:
        path: percorso del file YAML da scrivere.
        data_path: cartella del mini-dataset.
        backbone: backbone da verificare ("resnet" o "videomae").
        device: dispositivo di calcolo ("cpu" o "cuda").
        method: metodo da verificare ("full" o "frozen").
        features_path: percorso della cache feature del run (solo per
            "frozen"; distinto per run, così ogni run estrae le proprie
            feature e la verifica copre anche l'estrazione).
    """
    common = (
        "run_name: placeholder\n"
        f"backbone: {backbone}\n"
        f"data_path: {data_path}\n"
        "num_clusters: 10\n"
        "max_iterations: 2\n"
        "stability_threshold: 1.1\n"
        "min_cluster_size: 1\n"
        "epochs: 1\n"
        "lr_head: 1.0e-3\n"
        "weight_decay: 1.0e-4\n"
        "seed: 42\n"
        f"device: {device}\n"
    )
    if method == "full":
        specific = (
            "num_blocks: 2\n"
            "batch_size: 4\n"
            "lr_backbone: 1.0e-4\n"
        )
    else:
        specific = (
            f"features_path: {features_path}\n"
            "projection_dim: 256\n"
            "extract_batch_size: 4\n"
            "batch_size: 8\n"
            "lr_projection: 1.0e-4\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write(common + specific)


def compare_runs() -> bool:
    """
    Confronta gli artefatti dei due run.

    Returns:
        True se storico, assegnazioni e checkpoint sono identici.
    """
    all_ok = True

    histories = []
    for run_name in RUN_NAMES:
        with open(os.path.join(LOGS_ROOT, run_name, "history.json"), encoding="utf-8") as f:
            histories.append(json.load(f))
    identical = histories[0] == histories[1]
    print(f"history.json identici: {identical}")
    all_ok = all_ok and identical

    num_iterations = len(histories[0])
    for iteration in range(num_iterations):
        filename = f"assignments_iter{iteration:02d}.npy"
        assignments = [
            np.load(os.path.join(LOGS_ROOT, run_name, filename)) for run_name in RUN_NAMES
        ]
        identical = bool(np.array_equal(assignments[0], assignments[1]))
        print(f"{filename} identici: {identical}")
        all_ok = all_ok and identical

    state_dicts = [
        torch.load(os.path.join(CHECKPOINTS_ROOT, run_name + ".pt"), weights_only=True)
        for run_name in RUN_NAMES
    ]
    mismatched = [
        key for key in state_dicts[0] if not torch.equal(state_dicts[0][key], state_dicts[1][key])
    ]
    print(
        f"checkpoint identici: {not mismatched} "
        f"({len(state_dicts[0])} parametri confrontati"
        + (f", differenze in: {mismatched[:5]}" if mismatched else "")
        + ")"
    )
    all_ok = all_ok and not mismatched

    return all_ok


def cleanup_runs() -> None:
    """Rimuove gli artefatti dei run di verifica."""
    for run_name in RUN_NAMES:
        shutil.rmtree(os.path.join(LOGS_ROOT, run_name), ignore_errors=True)
        checkpoint = os.path.join(CHECKPOINTS_ROOT, run_name + ".pt")
        if os.path.exists(checkpoint):
            os.remove(checkpoint)


def main() -> int:
    """Esegue i due mini-run e confronta gli artefatti."""
    parser = argparse.ArgumentParser(description="Verifica del determinismo della pipeline iterativa")
    parser.add_argument(
        "--backbone",
        default="resnet",
        choices=["resnet", "videomae"],
        help="backbone da verificare",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="dispositivo di calcolo (cpu o cuda)",
    )
    parser.add_argument(
        "--method",
        default="full",
        choices=["full", "frozen"],
        help="metodo da verificare: loop completo o backbone congelato",
    )
    args = parser.parse_args()

    entry_point = (
        "src.training.run_iterative" if args.method == "full" else "src.training.run_iterative_frozen"
    )

    temp_dir = tempfile.mkdtemp(prefix="determinism_check_")
    mini_data = os.path.join(temp_dir, "mini_data")

    try:
        num_videos = build_mini_dataset("data", mini_data)
        print(
            f"Mini-dataset: {num_videos} video | backbone: {args.backbone} | "
            f"device: {args.device} | metodo: {args.method}"
        )

        for run_name in RUN_NAMES:
            # Configurazione distinta per run: nel metodo frozen ogni run
            # deve estrarre le proprie feature (features_path separati)
            config_path = os.path.join(temp_dir, f"check_config_{run_name}.yaml")
            features_path = os.path.join(temp_dir, f"features_{run_name}.pt").replace("\\", "/")
            write_check_config(
                config_path, mini_data, args.backbone, args.device, args.method, features_path
            )

            print(f"Esecuzione mini-run '{run_name}'...")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    entry_point,
                    "--config",
                    config_path,
                    "--run-name",
                    run_name,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Mini-run '{run_name}' fallito:\n{result.stderr}")
                return 1

        all_ok = compare_runs()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if all_ok:
        cleanup_runs()
        print("VERIFICA SUPERATA: due run identici, la pipeline è deterministica")
        return 0

    print("VERIFICA FALLITA: i run differiscono, c'è una sorgente di casualità non seedata")
    print(f"(artefatti lasciati in {LOGS_ROOT} per l'ispezione)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Materializzazione dei sottoinsiemi storici del dataset (398 e 1.865).

Il dataset in ``data/`` e' cresciuto in tre fasi (398 -> 1.865 -> 5.802
video) e i run storici sono stati eseguiti sulle prime due istantanee,
che questo script ricostruisce come cartelle autonome:

- ``data/subsets/n398``: i video originali, identificati dalla data di
  modifica dei file (le estensioni del dataset sono tutte successive
  al cutoff); stesso criterio gia' verificato bit-esatto in
  ``src.utils.plot_tsne``;
- ``data/subsets/n1865``: i 398 originali piu' tutti i video degli
  split ufficiali val/test di Kinetics-400, identificati in modo esatto
  tramite i CSV ufficiali conservati in ``data/_kinetics_tmp/`` (i nomi
  dei file sono gli youtube_id).

I file vengono materializzati come hardlink (zero spazio disco extra;
copia come ripiego), preservando la struttura ``<classe>/<video>.mp4``:
cosi' la cache dei frame esistente viene riutilizzata invariata e i
sottoinsiemi sono utilizzabili come ``data_path`` da qualunque run.

Ogni sottoinsieme viene verificato dopo la materializzazione: la
distribuzione per classe del 398 deve coincidere esattamente con quella
nota, il totale del 1865 deve essere 1.865 e nessun video originale
deve comparire negli split val/test. In caso di verifica fallita lo
script esce con codice diverso da zero.

Uso (dalla radice del repository):
    python -m src.datasets.materialize_subsets
"""

import csv
import os
import shutil
import sys
from datetime import datetime

SUBSETS_ROOT = os.path.join("data", "subsets")
KINETICS_TMP = os.path.join("data", "_kinetics_tmp")
# I video originali sono stati scaricati prima di questa data; le
# estensioni del dataset (val/test/train di Kinetics) sono successive
ORIGINAL_CUTOFF = datetime(2026, 7, 5).timestamp()
# Distribuzione per classe attesa dei 398 originali (ordine alfabetico),
# la stessa verificata in src.utils.plot_tsne
EXPECTED_COUNTS_398 = [45, 38, 35, 35, 34, 43, 44, 42, 37, 45]
EXPECTED_TOTAL_1865 = 1865


def load_val_test_ids() -> set[str]:
    """
    Carica gli youtube_id degli split ufficiali val e test.

    Returns:
        Insieme degli id presenti nei CSV ufficiali val.csv e test.csv.
    """
    ids: set[str] = set()
    for split_csv in ("val.csv", "test.csv"):
        with open(os.path.join(KINETICS_TMP, split_csv), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids.add(row["youtube_id"])
    return ids


def classify_videos(data_root: str = "data") -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Classifica i video correnti nei due sottoinsiemi storici.

    Args:
        data_root: cartella radice del dataset completo.

    Returns:
        Coppia (originali, val_test): liste di tuple (classe, nome file).
        Il sottoinsieme 1865 e' l'unione delle due liste.

    Raises:
        RuntimeError: se un video originale compare negli split val/test
            (indizio di identificazione errata).
    """
    val_test_ids = load_val_test_ids()
    originals: list[tuple[str, str]] = []
    val_test: list[tuple[str, str]] = []

    for class_dir in sorted(os.listdir(data_root)):
        full_dir = os.path.join(data_root, class_dir)
        if not os.path.isdir(full_dir) or class_dir in ("frame_cache", "subsets") or class_dir.startswith("_"):
            continue
        for video in sorted(os.listdir(full_dir)):
            if not video.endswith("mp4"):
                continue
            path = os.path.join(full_dir, video)
            if os.path.getmtime(path) < ORIGINAL_CUTOFF:
                if video[:-4] in val_test_ids:
                    raise RuntimeError(f"Video originale presente in val/test: {path}")
                originals.append((class_dir, video))
            elif video[:-4] in val_test_ids:
                val_test.append((class_dir, video))

    return originals, val_test


def materialize(name: str, videos: list[tuple[str, str]], data_root: str = "data") -> str:
    """
    Materializza un sottoinsieme come cartella di hardlink.

    Args:
        name: nome della cartella del sottoinsieme (es. "n398").
        videos: lista di tuple (classe, nome file) da includere.
        data_root: cartella radice del dataset completo.

    Returns:
        Percorso della cartella materializzata.
    """
    target_root = os.path.join(SUBSETS_ROOT, name)
    num_linked = 0
    for class_dir, video in videos:
        source = os.path.join(data_root, class_dir, video)
        target_dir = os.path.join(target_root, class_dir)
        target = os.path.join(target_dir, video)
        os.makedirs(target_dir, exist_ok=True)
        if os.path.exists(target):
            continue
        try:
            os.link(source, target)
        except OSError:
            # Ripiego per filesystem senza supporto hardlink
            shutil.copy2(source, target)
        num_linked += 1

    print(f"{target_root}: {len(videos)} video ({num_linked} materializzati ora)")
    return target_root


def verify_subset(root: str, expected_counts: list[int] | None, expected_total: int) -> bool:
    """
    Verifica la composizione di un sottoinsieme materializzato.

    Args:
        root: cartella del sottoinsieme.
        expected_counts: distribuzione per classe attesa (ordine
            alfabetico), o None per verificare solo il totale.
        expected_total: numero totale di video atteso.

    Returns:
        True se la verifica passa.
    """
    counts = []
    for class_dir in sorted(os.listdir(root)):
        full_dir = os.path.join(root, class_dir)
        if os.path.isdir(full_dir):
            counts.append(sum(1 for v in os.listdir(full_dir) if v.endswith("mp4")))

    total = sum(counts)
    ok = total == expected_total and (expected_counts is None or counts == expected_counts)
    print(f"{root}: totale {total} (atteso {expected_total}), distribuzione {counts} -> "
          f"{'OK' if ok else 'ERRORE'}")
    return ok


def main() -> int:
    """Materializza e verifica i due sottoinsiemi storici."""
    originals, val_test = classify_videos()
    print(f"Identificati: {len(originals)} originali, {len(val_test)} da val/test")

    root_398 = materialize("n398", originals)
    root_1865 = materialize("n1865", originals + val_test)

    all_ok = verify_subset(root_398, EXPECTED_COUNTS_398, sum(EXPECTED_COUNTS_398))
    all_ok = verify_subset(root_1865, None, EXPECTED_TOTAL_1865) and all_ok

    if all_ok:
        print("VERIFICA SUPERATA: sottoinsiemi materializzati correttamente")
        return 0

    print("VERIFICA FALLITA: composizione dei sottoinsiemi non corrispondente")
    return 1


if __name__ == "__main__":
    sys.exit(main())

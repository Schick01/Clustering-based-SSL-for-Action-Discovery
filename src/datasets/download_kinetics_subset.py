"""
Harvesting di un sottoinsieme di Kinetics-400 dagli archivi ufficiali.

Scarica in streaming le parti tar.gz degli split richiesti dal mirror
S3 della CVD Foundation (https://github.com/cvdfoundation/kinetics-dataset),
estrae solo i video delle classi già presenti in ``data/`` e li salva
come ``data/<classe>/<youtube_id>.mp4``, stessa convenzione del dataset
esistente. Ogni archivio viene eliminato subito dopo l'estrazione: il
picco di occupazione disco resta nell'ordine della singola parte
(~2 GB), indipendentemente dal traffico totale.

Il lavoro è riprendibile: le parti completate vengono annotate in un
file di stato dentro la cartella di lavoro e saltate ai run successivi,
quindi un'interruzione (rete, riavvio) non fa perdere il progresso.

Uso (dalla radice del repository):
    python -m src.datasets.download_kinetics_subset --splits val test
    python -m src.datasets.download_kinetics_subset --splits val --max-parts 1   # collaudo
"""

import argparse
import csv
import json
import os
import re
import tarfile

import requests

S3_BASE = "https://s3.amazonaws.com/kinetics/400"
# Nome dei file negli archivi: <youtube_id>_<inizio:06d>_<fine:06d>.mp4
VIDEO_NAME_PATTERN = re.compile(r"^(.+)_\d{6}_\d{6}\.mp4$")
DOWNLOAD_CHUNK_BYTES = 1 << 20


def list_target_classes(data_root: str) -> list[str]:
    """
    Elenca le classi del dataset locale (le sottocartelle di ``data/``).

    Args:
        data_root: cartella radice del dataset.

    Returns:
        Nomi delle classi, escluse la cache dei frame e le cartelle di
        lavoro (prefisso "_").
    """
    return sorted(
        entry
        for entry in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, entry))
        and entry != "frame_cache"
        and not entry.startswith("_")
    )


def download_file(url: str, destination: str) -> None:
    """
    Scarica un file in streaming su disco.

    Args:
        url: URL sorgente.
        destination: percorso di destinazione.
    """
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                f.write(chunk)


def load_id_to_class(work_dir: str, split: str, target_classes: list[str]) -> dict[str, str]:
    """
    Costruisce la mappa youtube_id -> classe per uno split, limitata
    alle classi di interesse, scaricando il CSV ufficiale se assente.

    Args:
        work_dir: cartella di lavoro (per la cache dei CSV).
        split: nome dello split ("train", "val" o "test").
        target_classes: classi da tenere.

    Returns:
        Dizionario youtube_id -> nome classe.
    """
    csv_path = os.path.join(work_dir, f"{split}.csv")
    if not os.path.exists(csv_path):
        download_file(f"{S3_BASE}/annotations/{split}.csv", csv_path)

    wanted = set(target_classes)
    id_to_class: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] in wanted:
                id_to_class[row["youtube_id"]] = row["label"]
    return id_to_class


def list_part_urls(work_dir: str, split: str) -> list[str]:
    """
    Recupera l'elenco degli URL delle parti tar.gz di uno split.

    Args:
        work_dir: cartella di lavoro (per la cache dell'elenco).
        split: nome dello split.

    Returns:
        Lista di URL, nell'ordine ufficiale.
    """
    list_path = os.path.join(work_dir, f"k400_{split}_path.txt")
    if not os.path.exists(list_path):
        download_file(f"{S3_BASE}/{split}/k400_{split}_path.txt", list_path)

    with open(list_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def count_videos_per_class(data_root: str, target_classes: list[str]) -> dict[str, int]:
    """
    Conta i video attualmente presenti per ciascuna classe.

    Args:
        data_root: cartella radice del dataset.
        target_classes: classi da contare.

    Returns:
        Dizionario classe -> numero di video.
    """
    return {
        class_name: len(
            [v for v in os.listdir(os.path.join(data_root, class_name)) if v.endswith(".mp4")]
        )
        for class_name in target_classes
    }


def harvest_part(
    part_url: str,
    work_dir: str,
    id_to_class: dict[str, str],
    data_root: str,
) -> int:
    """
    Scarica una parte, estrae i video delle classi di interesse e la elimina.

    I video già presenti in ``data/`` non vengono sovrascritti.

    Args:
        part_url: URL della parte tar.gz.
        work_dir: cartella di lavoro per il tar temporaneo.
        id_to_class: mappa youtube_id -> classe dello split corrente.
        data_root: cartella radice del dataset.

    Returns:
        Numero di video estratti da questa parte.
    """
    tar_path = os.path.join(work_dir, os.path.basename(part_url))
    download_file(part_url, tar_path)

    extracted = 0
    try:
        # Lettura sequenziale ("r|gz"): niente indice casuale, adatta a
        # scorrere una volta sola l'archivio compresso
        with tarfile.open(tar_path, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                match = VIDEO_NAME_PATTERN.match(os.path.basename(member.name))
                if match is None:
                    continue
                youtube_id = match.group(1)
                class_name = id_to_class.get(youtube_id)
                if class_name is None:
                    continue

                destination = os.path.join(data_root, class_name, youtube_id + ".mp4")
                if os.path.exists(destination):
                    continue

                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, open(destination, "wb") as out:
                    while True:
                        chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        out.write(chunk)
                extracted += 1
    finally:
        os.remove(tar_path)

    return extracted


def main() -> None:
    """Esegue l'harvesting degli split richiesti, in modo riprendibile."""
    parser = argparse.ArgumentParser(
        description="Scarica dai tar ufficiali di Kinetics-400 i video delle classi locali"
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--data-root", default="data", help="cartella radice del dataset")
    parser.add_argument("--max-parts", type=int, default=None, help="limite di parti per split (per collaudo)")
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="ferma l'harvesting quando ogni classe raggiunge questo numero di video",
    )
    args = parser.parse_args()

    work_dir = os.path.join(args.data_root, "_kinetics_tmp")
    os.makedirs(work_dir, exist_ok=True)
    state_path = os.path.join(work_dir, "state.json")

    state: dict = {"completed_parts": [], "extracted_total": 0}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)

    target_classes = list_target_classes(args.data_root)
    print(f"Classi di interesse ({len(target_classes)}): {target_classes}")

    for split in args.splits:
        id_to_class = load_id_to_class(work_dir, split, target_classes)
        part_urls = list_part_urls(work_dir, split)
        if args.max_parts is not None:
            part_urls = part_urls[: args.max_parts]
        print(f"\nSplit '{split}': {len(id_to_class)} video candidati, {len(part_urls)} parti da processare")

        for index, part_url in enumerate(part_urls):
            # Stop anticipato: tutte le classi hanno raggiunto il target
            if args.target_per_class is not None:
                counts = count_videos_per_class(args.data_root, target_classes)
                if min(counts.values()) >= args.target_per_class:
                    print(f"Target di {args.target_per_class} video/classe raggiunto: stop")
                    break

            part_name = f"{split}/{os.path.basename(part_url)}"
            if part_name in state["completed_parts"]:
                continue

            extracted = harvest_part(part_url, work_dir, id_to_class, args.data_root)
            state["completed_parts"].append(part_name)
            state["extracted_total"] += extracted
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            print(
                f"[{split} {index + 1}/{len(part_urls)}] {os.path.basename(part_url)}: "
                f"+{extracted} video (totale estratti: {state['extracted_total']})",
                flush=True,
            )

    print("\nRiepilogo finale per classe:")
    for class_name in target_classes:
        count = len(
            [v for v in os.listdir(os.path.join(args.data_root, class_name)) if v.endswith(".mp4")]
        )
        print(f"  {class_name}: {count} video")


if __name__ == "__main__":
    main()

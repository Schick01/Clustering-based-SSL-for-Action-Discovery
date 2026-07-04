"""
Verifica del caricamento dei pesi pre-addestrati di VideoMAE.

Il checkpoint ``MCG-NJU/videomae-base`` salva i bias di attention nel
formato ``q_bias``/``v_bias`` (query e value separati, key senza bias).
Alcune versioni di transformers (serie 5.x) non mappano queste chiavi e
reinizializzano i bias a zero, producendo silenziosamente un modello
diverso da quello pre-addestrato. Questo script verifica che:

1. i bias di attention del modello caricato non siano tutti nulli;
2. coincidano esattamente con i valori nel file del checkpoint
   scaricato dall'hub, per tutti i layer dell'encoder.

Uso (dalla radice del repository):
    python -m tests.verify_videomae_weights
"""

import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import VideoMAEModel

MODEL_NAME = "MCG-NJU/videomae-base"
# Prefisso delle chiavi nel checkpoint di pre-training rispetto al
# modello encoder-only caricato con VideoMAEModel
CHECKPOINT_PREFIX = "videomae."


def main() -> int:
    """Confronta i bias di attention del modello con il checkpoint."""
    import transformers

    print(f"transformers: {transformers.__version__}")
    print(f"Caricamento modello e checkpoint '{MODEL_NAME}'...")

    model = VideoMAEModel.from_pretrained(MODEL_NAME)
    checkpoint = load_file(hf_hub_download(MODEL_NAME, "model.safetensors"))

    num_layers = len(model.encoder.layer)
    all_ok = True

    for layer_idx in range(num_layers):
        attention = model.encoder.layer[layer_idx].attention.attention

        for bias_name in ("q_bias", "v_bias"):
            model_bias = getattr(attention, bias_name, None)
            if model_bias is None:
                print(f"layer {layer_idx:2d} {bias_name}: ASSENTE nel modello (implementazione incompatibile) [FAIL]")
                all_ok = False
                continue

            checkpoint_key = (
                f"{CHECKPOINT_PREFIX}encoder.layer.{layer_idx}.attention.attention.{bias_name}"
            )
            checkpoint_bias = checkpoint[checkpoint_key]

            nonzero = model_bias.abs().max().item() > 0
            identical = torch.equal(model_bias.detach(), checkpoint_bias)

            if not (nonzero and identical):
                print(
                    f"layer {layer_idx:2d} {bias_name}: non nullo = {nonzero}, "
                    f"identico al checkpoint = {identical} [FAIL]"
                )
                all_ok = False

    if all_ok:
        print(f"VERIFICA SUPERATA: q_bias/v_bias fedeli al checkpoint su tutti i {num_layers} layer")
        return 0

    print("VERIFICA FALLITA: i pesi pre-addestrati NON sono caricati correttamente")
    print("(controllare la versione di transformers: richiesta la serie 4.x, vedi environment.yml)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
#SBATCH --job-name=frozen-videomae
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-large
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:11264
#SBATCH --nodelist=gnode10
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lorenzocomis31@gmail.com
#SBATCH --output=logs/job-%j.log

# Metodo a backbone congelato su VideoMAE (dataset pieno), device CUDA.
# Prima il cancello di determinismo del percorso frozen, poi il run:
# se il cancello fallisce il run non parte (disciplina dei cancelli).
# Argomenti opzionali: $1 = config (default: quella ufficiale),
# $2 = nome del run (default: quello della config).

CONFIG=${1:-experiments/configs/frozen_videomae.yaml}
RUN_NAME_ARG=""
[ -n "$2" ] && RUN_NAME_ARG="--run-name $2"

mkdir -p logs
export HF_HUB_OFFLINE=1

echo "=== GATE DETERMINISMO FROZEN: $(date) su $(hostname) ==="
apptainer run --nv /shared/sifs/latest.sif python -m tests.verify_determinism \
    --backbone videomae --device cuda --method frozen
[ $? -eq 0 ] || { echo "=== GATE FALLITO: $(date) ==="; exit 1; }

echo "=== RUN FROZEN VIDEOMAE ($CONFIG $2): $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer run --nv /shared/sifs/latest.sif python -m src.training.run_iterative_frozen \
    --config "$CONFIG" --device cuda $RUN_NAME_ARG

[ $? -eq 0 ] && echo "=== RUN COMPLETATO: $(date) ===" || { echo "=== RUN FALLITO: $(date) ==="; exit 1; }

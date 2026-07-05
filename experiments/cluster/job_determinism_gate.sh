#!/bin/bash
#SBATCH --job-name=iterclust-gate
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-medium
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 --gres=shard:5632
#SBATCH --nodelist=gnode10
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lorenzocomis31@gmail.com
#SBATCH --output=logs/job-%j.log

# Cancello di determinismo su GPU: due mini-run del loop devono produrre
# artefatti identici, per entrambi i backbone, prima del run lungo.
# HF_HUB_OFFLINE forza l'uso della cache locale dei modelli (i nodi
# hanno DNS inaffidabile dentro il container).

mkdir -p logs
export HF_HUB_OFFLINE=1

echo "=== GATE DETERMINISMO: $(date) su $(hostname) ==="

echo "--- backbone resnet ---"
apptainer run --nv /shared/sifs/latest.sif python -m tests.verify_determinism --backbone resnet --device cuda
RESNET_OK=$?

echo "--- backbone videomae ---"
apptainer run --nv /shared/sifs/latest.sif python -m tests.verify_determinism --backbone videomae --device cuda
VIDEOMAE_OK=$?

if [ $RESNET_OK -eq 0 ] && [ $VIDEOMAE_OK -eq 0 ]; then
    echo "=== GATE SUPERATO: $(date) ==="
else
    echo "=== GATE FALLITO (resnet=$RESNET_OK, videomae=$VIDEOMAE_OK): $(date) ==="
    exit 1
fi

#!/bin/bash
#SBATCH --job-name=iterclust-resnet
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
# gpu-large: con il dataset esteso la sola lettura della cache frame
# (~14 GB) satura il limite di RAM del profilo gpu-medium
#SBATCH --qos=gpu-large
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:11264
#SBATCH --nodelist=gnode10
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lorenzocomis31@gmail.com
#SBATCH --output=logs/job-%j.log

# Run del clustering iterativo su ResNet18, device CUDA.
# Argomenti opzionali: $1 = percorso della config (default: regime
# gentile), $2 = nome del run (default: quello della config).

CONFIG=${1:-experiments/configs/iterative_resnet_gentle.yaml}
RUN_NAME_ARG=""
[ -n "$2" ] && RUN_NAME_ARG="--run-name $2"

mkdir -p logs
export HF_HUB_OFFLINE=1

echo "=== RUN RESNET ($CONFIG $2): $(date) su $(hostname) ==="

apptainer run --nv /shared/sifs/latest.sif python -m src.training.run_iterative \
    --config "$CONFIG" --device cuda $RUN_NAME_ARG

[ $? -eq 0 ] && echo "=== RUN COMPLETATO: $(date) ===" || { echo "=== RUN FALLITO: $(date) ==="; exit 1; }

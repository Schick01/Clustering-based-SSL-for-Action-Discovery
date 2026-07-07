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

# Run del clustering iterativo su ResNet18 in regime gentile (config
# dell'ablation, device CUDA). Primo argomento opzionale: nome del run
# (default: quello della config), utile per distinguere run su versioni
# diverse del dataset senza sovrascrivere gli artefatti.

RUN_NAME_ARG=""
[ -n "$1" ] && RUN_NAME_ARG="--run-name $1"

mkdir -p logs
export HF_HUB_OFFLINE=1

echo "=== RUN RESNET GENTILE ($1): $(date) su $(hostname) ==="

apptainer run --nv /shared/sifs/latest.sif python -m src.training.run_iterative \
    --config experiments/configs/iterative_resnet_gentle.yaml --device cuda $RUN_NAME_ARG

[ $? -eq 0 ] && echo "=== RUN COMPLETATO: $(date) ===" || { echo "=== RUN FALLITO: $(date) ==="; exit 1; }

#!/bin/bash
#SBATCH --job-name=iterclust-videomae
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-large
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:11264
#SBATCH --nodelist=gnode10
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lorenzocomis31@gmail.com
#SBATCH --output=logs/job-%j.log

# Run completo del clustering iterativo su VideoMAE (config ufficiale,
# device CUDA). HF_HUB_OFFLINE forza l'uso della cache locale dei
# modelli (i nodi hanno DNS inaffidabile dentro il container).

mkdir -p logs
export HF_HUB_OFFLINE=1

echo "=== RUN VIDEOMAE: $(date) su $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer run --nv /shared/sifs/latest.sif python -m src.training.run_iterative \
    --config experiments/configs/iterative_videomae.yaml --device cuda

[ $? -eq 0 ] && echo "=== RUN COMPLETATO: $(date) ===" || { echo "=== RUN FALLITO: $(date) ==="; exit 1; }

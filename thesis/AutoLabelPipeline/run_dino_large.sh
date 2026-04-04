#!/bin/bash
#SBATCH --job-name=dino_large_extract
#SBATCH --output=logs/dino_extract/large_extract_%j.out
#SBATCH --error=logs/dino_extract/large_extract_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

source /gpfs/helios/home/dwenger/vlm_env/bin/activate
cd /gpfs/helios/home/dwenger/AutoLabelPipeline

# Run Stage 1 using the NEW large config
python -m signal_classifier.extract_features --config signal_classifier/config_large.yaml

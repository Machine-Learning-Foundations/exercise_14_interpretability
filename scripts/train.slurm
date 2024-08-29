#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=A40devel
#SBATCH --job-name=tr_net
#SBATCH --output=./out/train_net-%j.out
#SBATCH --error=./out/train_net-%j.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
echo "training network..."

module load Anaconda3
pip install -r requirements.txt

python ./src/deepfake_interpretation.py
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=18:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=train_pm
#SBATCH --output=slurm_pm_%j.out

conda activate /scratch/ab9738/fml/project/env_fmlproj/;
export PATH=/scratch/ab9738/fml/project/env_fmlproj/bin:$PATH;
cd /scratch/ab9738/fml/project/Rethinking-Clustering-for-Robustness
python main_clustr.py --checkpoint clustr_acttrades --pretrained-path pretrained_weights/resnet18.pt --epochs 25 --consistency-lambda 8 --actual-trades=True

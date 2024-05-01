#!/bin/bash

#SBATCH -p academic
#SBATCH --job-name=frcsyn-train_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --error=slurm_train_baseline_%A.err
#SBATCH --output=slurm_train_baseline_%A.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

# devices represents nstasks per node i guess
# 2 gpus are too many i guess

# module load cuda/11.8.0 # cuda/11.8
# module load cudnn8.5-cuda11.7/8.5.0.96 # cudnn/8.4.0.27-11.6--gcc--11.3.0
module load python/3.10.12 py-pip # python/3.10.8--gcc--11.3.0
pip install -r $workPath/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
workPath="/home/rpblair/Digital_Image_Processing/frcsyn-master"

wandb login

python $workPath/main.py fit -c $workPath/experiments/train.yml -c $workPath/experiments/synth.yml --data.datasets_root $workPath/data2/ --trainer.devices 1 --trainer.strategy ddp

#!/bin/bash

#SBATCH --job-name=frcsyn-align_faces
#SBATCH -p academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --error=slurm_align_faces_%A_%a.err
#SBATCH --array=0-4
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

set -e

# CASIA-WebFace and DCFace are ready to use without any preprocessing
echo $SLURM_ARRAY_TASK_ID
datasets=("Real/AgeDB" "Real/BUPT-BalancedFace" "Real/CFP-FP" "Real/ROF" "Synth/GANDiffFace")
image_size=112
margin=12
input="/home/rpblair/Digital_Image_Processing/frcsyn-master/data/${datasets[$SLURM_ARRAY_TASK_ID]}"
output="/home/rpblair/Digital_Image_Processing/frcsyn-master/data-${image_size}-m${margin}-aligned/${datasets[$SLURM_ARRAY_TASK_ID]}"

# input="$WORK/frcsyn-datasets/${datasets[$SLURM_ARRAY_TASK_ID]}"
# output="$WORK/frcsyn-datasets-${image_size}-m${margin}-aligned/${datasets[$SLURM_ARRAY_TASK_ID]}"

# mkdir -p $output

# module load profile/deeplrn # not available need to investigate
module load cuda/11.8.0 # cuda/11.8
module load cudnn8.5-cuda11.7/8.5.0.96 # cudnn/8.4.0.27-11.6--gcc--11.3.0
module load python/3.10.12 # python/3.10.8--gcc--11.3.0
module load py-pip

cd /home/rpblair/Digital_Image_Processing/frcsyn-master
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip3 install -r requirements.txt
python3 ./align_faces.py --input "$input" --output "$output" -r -m $margin -s $image_size -n 0 --allow-no-faces
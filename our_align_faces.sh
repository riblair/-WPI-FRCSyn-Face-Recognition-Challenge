#!/bin/bash

#SBATCH --job-name=frcsyn-align_faces
#SBATCH -p academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --error=our_align%A_%a.err
#SBATCH --array=0-8
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

set -e

# CASIA-WebFace and DCFace are ready to use without any preprocessing
echo $SLURM_ARRAY_TASK_ID
datasets=(  "Synth/GANDiffFace/Black_Male" "Synth/GANDiffFace/Asian_Male" "Synth/GANDiffFace/Black_Female" 
            "Synth/GANDiffFace/Indian_Female" "Synth/GANDiffFace/Indian_Male" "Synth/GANDiffFace/Other_Male" 
            "Synth/GANDiffFace/Other_Female" "Synth/GANDiffFace/White_Male" "Synth/GANDiffFace/White_Female")
image_size=112
margin=12
input="/home/rpblair/Digital_Image_Processing/frcsyn-master/data/${datasets[$SLURM_ARRAY_TASK_ID]}"
output="/home/rpblair/Digital_Image_Processing/frcsyn-master/data2/${datasets[$SLURM_ARRAY_TASK_ID]}"

mkdir -p $output

module load cuda/11.8.0 # cuda/11.8
module load cudnn8.5-cuda11.7/8.5.0.96 # cudnn/8.4.0.27-11.6--gcc--11.3.0
module load python/3.10.12 # python/3.10.8--gcc--11.3.0
module load py-pip

cd /home/rpblair/Digital_Image_Processing/frcsyn-master
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip3 -v install -r requirements.txt
python3 -u ./retina_face_script.py --input "$input" --output "$output"
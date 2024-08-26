#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -cwd
#$ -l h_vmem=64G
#$ -l h_rt=12:00:00
#$ -q gpu
#$ -pe gpu-a100 1

# Initialise the environment modules and load CUDA
. /etc/profile.d/modules.sh
module load cuda/12.1.1
module load python/3.11.4

# Activate Python environment
source /exports/eddie/scratch/s2601126/env_T/bin/activate

# Change to the script directory
cd MSc_D/TraderT/TS2S

# Define pairs of parameters for training
PARAMS=("1 1" "1 5" "1 10" "1 20" "5 1" "5 5" "5 10" "5 20" "10 1" "10 5" "10 10" "10 20")

# Run the Python script with parameters based on the task ID
# Ensure that SGE_TASK_ID is 1-based and maps correctly to zero-based array indexing
python TS2S_train.py ${PARAMS[$SGE_TASK_ID-1]}
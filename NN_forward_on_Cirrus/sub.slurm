#!/bin/bash
#
#SBATCH --job-name=NN_f
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00

#SBATCH --account=tc065-s2601126

# Load the required modules
module load pytorch/1.13.1-gpu

cd MSc_D/TraderT/NN_forward
srun python NN_f_cuda.py 10 10
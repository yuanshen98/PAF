#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=parser
#SBATCH --output=parser.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate env2

python parser.py ./Data/Data ./


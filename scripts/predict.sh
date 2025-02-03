#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=250GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2

# Specify the Conda environment name
ENV_NAME="thesis_env"

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate "$ENV_NAME"

# Run Python code
python3 src/predit.py

# Deactivate the Conda environment
conda deactivate

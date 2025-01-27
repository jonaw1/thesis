#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=8GB

# Specify the Conda environment name
ENV_NAME="myenv"

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Conda and try again."
    exit 1
fi

# Check if the Conda environment exists
if ! conda info --envs | grep -q "^$ENV_NAME"
then
    echo "Conda environment '$ENV_NAME' does not exist. Creating it using 'env.yml'..."
    
    conda env create -f env.yml
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# Activate the Conda environment
echo "Activating the Conda environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Deactivate the Conda environment
conda deactivate

#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=8GB

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

deactivate

#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=8GB

# Check if Python 3.9 is installed
if ! command -v python3.9 &> /dev/null
then
    echo "Python 3.9 could not be found. Installing Python 3.9..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.9 python3.9-venv
fi

# Check if the virtual environment already exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.9..."
    python3.9 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate
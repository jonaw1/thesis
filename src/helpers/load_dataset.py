import os
from .logger_config import logger
import requests
import json

DATA_DIR = os.path.join("src", "datasets")
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
COUNTERFACT_PATH = os.path.join(DATA_DIR, "counterfact.json")


def load_dataset():
    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {DATA_DIR}")

    # Download the dataset if it doesn't already exist
    if not os.path.exists(COUNTERFACT_PATH):
        logger.info(
            f"Counterfact dataset not found at {COUNTERFACT_PATH}. "
            + "Downloading..."
        )
        response = requests.get(COUNTERFACT_URL)
        with open(COUNTERFACT_PATH, "wb") as f:
            f.write(response.content)
        logger.info(f"Counterfact dataset saved to {COUNTERFACT_PATH}")
    else:
        logger.info(
            f"Counterfact dataset found at {COUNTERFACT_PATH}. "
            + "Skipping download."
        )

    with open(COUNTERFACT_PATH, "r") as f:
        counterfact = json.load(f)

    return counterfact

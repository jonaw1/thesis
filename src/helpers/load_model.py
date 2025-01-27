import os
from .logger_config import logger
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = os.path.join("src", "models", "gpt2-xl")


def load_model_and_tokenizer(device):
    # Create the model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {MODEL_DIR}")

    # Download the model if it doesn't already exist
    if not os.listdir(MODEL_DIR):  # Check if the directory is empty
        logger.info(f"Model not found at {MODEL_DIR}. Downloading...")
        model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        logger.info(f"Model and tokenizer saved to {MODEL_DIR}")
    else:
        logger.info(f"Model found at {MODEL_DIR}. Skipping download.")

    # Config tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Move the model to the selected device
    model = model.to(device)

    return model, tokenizer

from helpers import (
    get_device,
    load_model_and_tokenizer,
    load_dataset,
    logger,
)
import os
import json
import torch
import numpy as np
import random
import re
from datetime import datetime
import torch.nn.functional as F


NUM_ITERATIONS = 10

SUCCESSFUL_EDITS_PATH = os.path.join(
    "src", "other_cache", "successful_edits.json"
)
PROMPTS_PATH = os.path.join("src", "prompts", "prompts.json")
EDITED_MODELS_DIR = os.path.join(
    "src", "models", "edited", "rome", "counterfact"
)
MAX_NEW_TOKENS = 15
EXAMPLES_DIR = os.path.join("src", "regression_data", "rome", "counterfact")
now = datetime.now()
formatted_time = f"{now:%Y-%m-%d_%H:%M:%S},{int(now.microsecond / 1000):03d}"
EXAMPLES_PATH = os.path.join(EXAMPLES_DIR, f"{formatted_time}_gpt2-xl.json")
GENERATED_PATH = os.path.join(EXAMPLES_DIR, "generated.json")


def main():
    # Check if there are successful edits, abort if not
    if not os.path.exists(SUCCESSFUL_EDITS_PATH):
        logger.error(
            f"No successful edits found in {SUCCESSFUL_EDITS_PATH}. Aborting..."
        )
        return

    # Open successfull edits
    with open(SUCCESSFUL_EDITS_PATH, "r") as f:
        successful_edit_ids = json.load(f)
    num_successful_edits = len(successful_edit_ids)
    logger.info(
        f"Found {num_successful_edits} successful edits. Starting generation..."
    )

    # Load device, model, tokenizer and dataset
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device)
    counterfact = load_dataset()
    counterfact_len = len(counterfact)

    # Opening follow up prompts
    with open(PROMPTS_PATH, "r") as f:
        follow_up_prompts: dict = json.load(f)
    follow_up_prompts: list = [
        item for sublist in follow_up_prompts.values() for item in sublist
    ]
    num_follow_up_prompts = len(follow_up_prompts)
    logger.info(f"Found {num_follow_up_prompts} follow-up prompts.")

    # Opening already generated IDs
    if os.path.exists(GENERATED_PATH):
        with open(GENERATED_PATH, "r") as f:
            already_generated_ids = json.load(f)
        logger.info(
            f"Found {len(already_generated_ids)} already generated IDs."
        )
    else:
        already_generated_ids = []
        logger.info(
            f"No alredy generated IDs found. Initializing an empty list."
        )

    # Initialize results dictionary
    results = {}

    # Generate examples for each successful edit
    unedited_cf_ids = []
    results = {}
    x = 0
    for i, id in enumerate(successful_edit_ids):
        # If maximum number of iterations is reached, abort
        if x == NUM_ITERATIONS:
            break

        # If ID in already generated IDs, skip
        if id in already_generated_ids:
            continue

        x += 1

        logger.info(f"({x}/{NUM_ITERATIONS}) " + f"Generating examples...")

        # Updating edited model
        edited_weights_path = os.path.join(EDITED_MODELS_DIR, f"{id}.npz")
        logger.info(f"Loading edited weights from {edited_weights_path}...")
        loaded_params = np.load(edited_weights_path)
        params_e = loaded_params["arr"]
        logger.info(f"Applying edited weights to model...")

        # Convert the numpy array back to a PyTorch tensor
        params_e_tensor = torch.from_numpy(params_e).to(device)

        # Assign the loaded weights back to the edited model
        with torch.no_grad():
            model.transformer.h[17].mlp.c_proj.weight.copy_(params_e_tensor)

        logger.info(
            "Edited weights successfully loaded "
            + "and assigned to the edited model."
        )

        # Loading prompt from dataset
        cf = counterfact[id]
        edited_prompt = random.choice(cf["paraphrase_prompts"])
        edited_ground_truth = cf["requested_rewrite"]["target_true"]["str"]
        edited_target_new = cf["requested_rewrite"]["target_new"]["str"]
        logger.info(f"New cf (edited) with ID {id} loaded from the dataset")

        # Choose random counterfact that has not been used for editing
        unedited_cf_id = random.randint(0, counterfact_len - 1)
        while (
            unedited_cf_id in successful_edit_ids
            or unedited_cf_id in unedited_cf_ids
            or unedited_cf_id in already_generated_ids
        ):
            unedited_cf_id = random.randint(0, counterfact_len - 1)
        unedited_cf_ids.append(unedited_cf_id)
        logger.info(
            f"New random cf (unedited) with ID {unedited_cf_id} loaded from the dataset"
        )

        already_generated_ids.append(id)
        already_generated_ids.append(unedited_cf_id)

        # Loading unedited prompt from dataset
        unedited_cf = counterfact[unedited_cf_id]
        unedited_prompt = random.choice(unedited_cf["paraphrase_prompts"])
        unedited_ground_truth = unedited_cf["requested_rewrite"]["target_true"][
            "str"
        ]

        prompts = [unedited_prompt, edited_prompt]
        batch = tokenizer(prompts, return_tensors="pt", padding=True)

        logger.info("Generating outputs with edited model...")
        base_outputs = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        unedited_base_output = tokenizer.decode(
            base_outputs[0], skip_special_tokens=True
        )
        logger.info(
            "Unedited output "
            + f"({unedited_ground_truth} -> {unedited_ground_truth}): "
            + unedited_base_output,
        )

        edited_base_output = tokenizer.decode(
            base_outputs[1], skip_special_tokens=True
        )
        logger.info(
            f"Edited output ({edited_ground_truth} -> {edited_target_new}): "
            + edited_base_output,
        )

        unedited_results = []
        edited_results = []
        # Ask follow up questions and save results
        for j, question in enumerate(follow_up_prompts):
            logger.info(
                f"{x}/{NUM_ITERATIONS}_{j + 1}/{num_follow_up_prompts}: Generating results..."
            )
            unedited_prompt = f"{unedited_base_output}\nFollow-Up Question: {question}\nPlease answer with 'yes' or 'no':"
            edited_prompt = f"{edited_base_output}\nFollow-Up Question: {question}\nPlease answer with 'yes' or 'no':"
            prompts = [unedited_prompt, edited_prompt]
            batch = tokenizer(prompts, return_tensors="pt", padding=True)
            outputs = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                max_new_tokens=MAX_NEW_TOKENS,
            )
            max_length = batch["input_ids"].shape[-1]
            unedited_output = tokenizer.decode(
                outputs[0][max_length:], skip_special_tokens=True
            )
            edited_output = tokenizer.decode(
                outputs[1][max_length:], skip_special_tokens=True
            )

            # Count 'yes' and 'no' in the answers
            unedited_yes_count = len(
                re.findall(r"\byes\b", unedited_output, flags=re.IGNORECASE)
            )
            unedited_no_count = len(
                re.findall(r"\bno\b", unedited_output, flags=re.IGNORECASE)
            )

            if unedited_yes_count > unedited_no_count:
                unedited_results.append(1)
            elif unedited_no_count > unedited_yes_count:
                unedited_results.append(-1)
            else:
                unedited_results.append(0)

            edited_yes_count = len(
                re.findall(r"\byes\b", edited_output, flags=re.IGNORECASE)
            )
            edited_no_count = len(
                re.findall(r"\bno\b", edited_output, flags=re.IGNORECASE)
            )

            if edited_yes_count > edited_no_count:
                edited_results.append(1)
            elif edited_no_count > edited_yes_count:
                edited_results.append(-1)
            else:
                edited_results.append(0)

            # Log follow up results
            logger.info(f"ID: {id}")
            logger.info(f"Input: {edited_prompt}")
            logger.info(f"Output: {edited_output}")
            logger.info(f"Unedited ID: {unedited_cf_id}")
            logger.info(f"Input: {unedited_prompt}")
            logger.info(f"Output: {unedited_output}")

            logger.info(f"Getting top 10 token probabilities...")
            with torch.no_grad():
                logits_outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )

            logits = (
                logits_outputs.logits
            )  # Shape: (batch_size, seq_len, vocab_size)
            last_logits = logits[:, -1, :]  # Get the logits for the last token

            # Convert logits to probabilities
            probs = F.softmax(
                last_logits, dim=-1
            )  # Shape: (batch_size, vocab_size)

            # Get all token IDs and their corresponding tokens
            all_token_ids = torch.arange(
                probs.size(-1), device=device
            )  # All possible token IDs
            all_tokens = tokenizer.convert_ids_to_tokens(
                all_token_ids.tolist()
            )  # Convert IDs to tokens

            # Define a regex pattern to match valid words (letters only, no special characters or subword markers)
            valid_word_pattern = re.compile(r"^[a-zA-Z]+$")

            # Create a mask to filter out non-word tokens
            valid_token_mask = torch.tensor(
                [bool(valid_word_pattern.match(token)) for token in all_tokens],
                device=device,
            )

            # Apply the mask to the probabilities
            filtered_probs = probs * valid_token_mask.float()

            # Get the top 10 token probabilities from the filtered probabilities
            top_probs, top_indices = torch.topk(filtered_probs, k=10, dim=-1)

            # Convert token IDs to words
            top_words = [
                tokenizer.convert_ids_to_tokens(idx.tolist())
                for idx in top_indices
            ]

            # Print results
            for k, prompt in enumerate(prompts):
                logger.info(f"Prompt: {prompt}")
                for l in range(10):
                    logger.info(
                        f"Top {l+1}: {top_words[k][l]} (P={top_probs[k][l].item():.4f})"
                    )

        unedited_results.append(0)
        edited_results.append(1)

        results[id] = edited_results
        results[unedited_cf_id] = unedited_results

    # Create the model directory if it doesn't exist
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {EXAMPLES_DIR}")

    # Save already edited list
    with open(GENERATED_PATH, "w") as f:
        json.dump(already_generated_ids, f)
    logger.info(
        f"Already generated IDs saved to {GENERATED_PATH}."
        + f"There are now {len(already_generated_ids)} generated IDs."
    )

    # Save the results to a JSON file
    with open(EXAMPLES_PATH, "w") as f:
        json.dump(results, f)
    logger.info(f"Generated examples saved to {EXAMPLES_PATH}.")


if __name__ == "__main__":
    main()

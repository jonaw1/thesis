from helpers import get_device, load_model_and_tokenizer, load_dataset, logger
import os
import json
import torch
import numpy as np
import random

SUCCESSFUL_EDITS_PATH = os.path.join(
    "src", "other_cache", "successful_edits.json"
)
PROMPTS_PATH = os.path.join("src", "prompts", "prompts.json")
EDITED_MODELS_DIR = os.path.join(
    "src", "models", "edited", "rome", "counterfact"
)
MAX_NEW_TOKENS = 15
EXAMPLES_DIR = os.path.join("src", "regression_data", "rome", "counterfact")
EXAMPLES_PATH = os.path.join(EXAMPLES_DIR, "gpt2-xl.json")


def main():
    if not os.path.exists(SUCCESSFUL_EDITS_PATH):
        logger.error(
            f"No successful edits found in {SUCCESSFUL_EDITS_PATH}. Aborting..."
        )
        return

    if os.path.exists(EXAMPLES_PATH):
        logger.error(f"Results already exist in {EXAMPLES_PATH}. Aborting...")
        return

    # Load device, model, tokenizer and dataset
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device)
    counterfact = load_dataset()
    counterfact_len = len(counterfact)

    with open(SUCCESSFUL_EDITS_PATH, "r") as f:
        successful_edit_ids = json.load(f)

    num_successful_edits = len(successful_edit_ids)

    with open(PROMPTS_PATH, "r") as f:
        follow_up_prompts: dict = json.load(f)

    follow_up_prompts = [
        item for sublist in follow_up_prompts.values() for item in sublist
    ]

    unedited_cf_ids = []
    results = []
    for i, id in enumerate(successful_edit_ids):
        logger.info(
            f"({i + 1}/{num_successful_edits}) " + f"Generating examples..."
        )

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

        # Choose random counterfact that has not been used for editing
        unedited_cf_id = random.randint(0, counterfact_len - 1)
        while (
            unedited_cf_id in successful_edit_ids
            or unedited_cf_id in unedited_cf_ids
        ):
            unedited_cf_id = random.randint(0, counterfact_len - 1)
        unedited_cf_ids.append(unedited_cf_id)

        # Loading unedited prompt from dataset
        unedited_cf = counterfact[unedited_cf_id]
        unedited_prompt = random.choice(unedited_cf["paraphrase_prompts"])
        unedited_ground_truth = unedited_cf["requested_rewrite"]["target_true"]["str"]

        prompts = [unedited_prompt, edited_prompt]

        batch = tokenizer(prompts, return_tensors="pt", padding=True)

        logger.info("Generating base outputs...")
        base_outputs = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        logger.info(
            "Unedited output "
            + f"({unedited_ground_truth} -> {unedited_ground_truth}): "
            + tokenizer.decode(base_outputs[0], skip_special_tokens=True),
        )
        logger.info(
            f"Edited output ({edited_ground_truth} -> {edited_target_new}): "
            + tokenizer.decode(base_outputs[1], skip_special_tokens=True),
        )

        for j, base_outputs in enumerate(base_outputs):
            pass


if __name__ == "__main__":
    main()

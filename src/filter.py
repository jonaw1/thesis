from helpers import get_device, load_model_and_tokenizer, logger, load_dataset
import json
import os
import torch
import numpy as np

USED_IDS_FILE_PATH = os.path.join("src", "other_cache", "used_case_ids.json")
EXPECTED_NUM_EDITS = 1000
SUCCESSFUL_EDITS_PATH = os.path.join(
    "src", "other_cache", "successful_edits.json"
)
UNSUCCESSFUL_EDITS_PATH = os.path.join(
    "src", "other_cache", "unsuccessful_edits.json"
)
EDITED_MODELS_DIR = os.path.join(
    "src", "models", "edited", "rome", "counterfact"
)
MAX_NEW_TOKENS = 15


def main():
    if os.path.exists(SUCCESSFUL_EDITS_PATH):
        logger.error("Edits were already filtered. Aborting...")
        return

    if not os.path.exists(USED_IDS_FILE_PATH):
        logger.error(f"{USED_IDS_FILE_PATH} does not exist. Aborting...")
        return
    else:
        logger.info(
            "Loading counterfact IDs used for editing "
            + f"from {USED_IDS_FILE_PATH}..."
        )

    with open(USED_IDS_FILE_PATH, "r") as f:
        used_case_ids = json.load(f)

    if len(used_case_ids) != EXPECTED_NUM_EDITS:
        logger.error(
            f"There must be {EXPECTED_NUM_EDITS} counterfact IDs. "
            + "Aborting..."
        )
        return

    # Loading device, models, tokenzier and dataset
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device)
    edited_model, _ = load_model_and_tokenizer(device)
    counterfact = load_dataset()

    successful_edit_ids = []
    unsuccessful_edit_ids = []
    for j, id in enumerate(used_case_ids):
        cf = counterfact[id]
        original_prompt = cf["requested_rewrite"]["prompt"].format(
            cf["requested_rewrite"]["subject"]
        )
        paraphrase_prompts = cf["paraphrase_prompts"]
        ground_truth = cf["requested_rewrite"]["target_true"]["str"]
        target_new = cf["requested_rewrite"]["target_new"]["str"]

        batch = tokenizer(paraphrase_prompts, return_tensors="pt", padding=True)

        logger.info(
            f"({j + 1}/{EXPECTED_NUM_EDITS}) Analyzing counterfact "
            + f"with case id {id}..."
        )
        logger.info(
            f"Prompt: {original_prompt}... {ground_truth} -> {target_new}"
        )

        # Updating edited model
        edited_weights_path = os.path.join(EDITED_MODELS_DIR, f"{id}.npz")
        logger.info(f"Loading edited weights from {edited_weights_path}...")
        loaded_params = np.load(edited_weights_path)
        params_e = loaded_params["arr"]

        # Convert the numpy array back to a PyTorch tensor
        params_e_tensor = torch.from_numpy(params_e).to(device)

        # Assign the loaded weights back to the edited model
        with torch.no_grad():
            edited_model.transformer.h[17].mlp.c_proj.weight.copy_(
                params_e_tensor
            )

        logger.info(
            "Edited weights successfully loaded "
            + "and assigned to the edited model."
        )

        # Generate pre-edit outputs
        logger.info("Generating pre-edit outputs...")
        pre_edit_outputs = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
        )

        # Generate post-edit outputs
        logger.info("Generating post-edit outputs...")
        post_edit_outputs = edited_model.generate(
            input_ids=batch["input_ids"].to(edited_model.device),
            attention_mask=batch["attention_mask"].to(edited_model.device),
            max_new_tokens=MAX_NEW_TOKENS,
        )

        max_length = batch["input_ids"].shape[-1]
        successful = True
        for i in range(len(paraphrase_prompts)):
            logger.info(f"Paraphrase prompt: {paraphrase_prompts[i]}")
            logger.info(
                f"Pre-Edit Output: {tokenizer.decode( pre_edit_outputs[i][max_length:], skip_special_tokens=True)}"
            )
            post_edit_output = tokenizer.decode(
                post_edit_outputs[i][max_length:], skip_special_tokens=True
            )
            logger.info(f"Post-Edit Output: {post_edit_output}")
            if target_new not in post_edit_output:
                logger.info(
                    f"{target_new} not in output, discarding counterfact with case id {id}..."
                )
                successful = False
                unsuccessful_edit_ids.append(id)
                break
            logger.info("--" * 50)

        if successful:
            successful_edit_ids.append(id)
            logger.info(f"Counterfact with case id {id} successful!")

    # Save the successful and unsuccessful edit IDs to a JSON file
    logger.info("Saving successful and unsuccessful edit IDs to JSON files...")

    with open(SUCCESSFUL_EDITS_PATH, "w") as f:
        json.dump(successful_edit_ids, f)

    with open(UNSUCCESSFUL_EDITS_PATH, "w") as f:
        json.dump(unsuccessful_edit_ids, f)


if __name__ == "__main__":
    main()

import os
from helpers import (
    logger,
    load_dataset,
    download_model_if_needed,
    load_editor,
)
import json
import random
import numpy as np

NUM_EDITS_PER_EXECUTION = 5
EDITED_MODELS_DIR = os.path.join(
    "src", "models", "edited", "rome", "counterfact"
)
USED_IDS_FILE_PATH = os.path.join("src", "other_cache", "used_case_ids.json")


def main():
    # Download model if needed (necessary step for editor)
    download_model_if_needed()

    # Loading dataset and editor
    counterfact = load_dataset()
    counterfact_len = len(counterfact)
    editor = load_editor()

    # Loading used case ids from cache (if exists)
    if os.path.exists(USED_IDS_FILE_PATH):
        with open(USED_IDS_FILE_PATH, "r") as f:
            used_case_ids = json.load(f)
        logger.info(
            f"Loaded {len(used_case_ids)} used_case_ids "
            + "from {USED_IDS_FILE_PATH}."
        )
    else:
        used_case_ids: list[int] = []
        logger.info(
            f"{USED_IDS_FILE_PATH} not found. Initializing an empty list."
        )

    # Create output directory if it doesn't exist
    os.makedirs(EDITED_MODELS_DIR, exist_ok=True)
    logger.info(f"Ensured directories exist: {EDITED_MODELS_DIR}")

    for i in range(NUM_EDITS_PER_EXECUTION):
        # Find case id (fact) that has not been used for editing before
        random_case_id = random.randint(0, counterfact_len - 1)
        while random_case_id in used_case_ids:
            random_case_id = random.randint(0, counterfact_len - 1)

        used_case_ids.append(random_case_id)

        cf = counterfact[random_case_id]

        subjects = [cf["requested_rewrite"]["subject"]]
        prompts = [
            cf["requested_rewrite"]["prompt"].format(
                cf["requested_rewrite"]["subject"]
            )
        ]
        gt = [cf["requested_rewrite"]["target_true"]["str"]]
        tn = [cf["requested_rewrite"]["target_new"]["str"]]

        logger.info(
            f"({i + 1}/{NUM_EDITS_PER_EXECUTION}) "
            + f"Editing model for: {prompts[0]}... {gt[0]} -> {tn[0]}..."
        )
        _, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=gt,
            target_new=tn,
            subject=subjects,
            sequential_edit=True,
        )

        params_e = (
            edited_model.transformer.h[17]
            .mlp.c_proj.weight.detach()
            .cpu()
            .numpy()
        )
        params_e = params_e.astype(np.float32)

        np.savez_compressed(
            f"{EDITED_MODELS_DIR}/{random_case_id}.npz", arr=params_e
        )

    with open(USED_IDS_FILE_PATH, "w") as f:
        json.dump(used_case_ids, f)

    with open(USED_IDS_FILE_PATH, "r") as f:
        used_case_ids_loaded: list[int] = json.load(f)

    logger.info(
        f"Finished. There are now {len(used_case_ids)} ids "
        + f"in the list in memory and {len(used_case_ids_loaded)} in the file"
    )


if __name__ == "__main__":
    main()

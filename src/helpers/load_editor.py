from EasyEdit.easyeditor import ROMEHyperParams, BaseEditor
from .logger_config import logger
import os

HPARAMS_PATH = os.path.join(
    "src", "EasyEdit", "hparams", "ROME", "gpt2-xl.yaml"
)


def load_editor():
    # Load hyperparameters
    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)

    # Instantiate the editor
    logger.info("Instantiating the editor...")
    return BaseEditor.from_hparams(hparams)

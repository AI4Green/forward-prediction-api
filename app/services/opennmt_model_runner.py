import os

from fastapi.logger import logger
from onmt.translate.translation_server import ServerModel as ONMTServerModel


class OpenNMTModelRunner:
    def __init__(self, n_best=1, beam_size=1, use_cuda=False):
        self.n_best = n_best
        self.beam_size = beam_size
        self.use_cuda = use_cuda
        self.model = None

    def load(self, model_dir: str, model_filename: str):
        """
        Load the OpenNMT model from checkpoint (.pt file)

        Args:
            model_dir (str): Directory containing the model file
            model_filename (str): Specific model filename to load
        """
        checkpoint_file = os.path.join(model_dir, model_filename)
        logger.info(f"Loading model from {checkpoint_file}")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Model file not found: {checkpoint_file}")

        onmt_config = {
            "models": checkpoint_file,
            "n_best": self.n_best,
            "beam_size": self.beam_size,
        }

        self.model = ONMTServerModel(opt=onmt_config, model_id=0, load=True)
        logger.info("ONMT model loaded successfully.")

    def run(self, data):
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        return self.model.run(data)

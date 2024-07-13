import sys
from utils.enums import SetType

from dataset.tinystories_dataset import TinyStoriesDataset
from torch.utils.data import DataLoader

from



class Trainer():
    def __init__(self, config):
        self.config = config

        self._prepare_data()

    def _prepare_data(self):
        self.dataset = getattr(sys.modules[__name__], self.config.data_cfg.name)

        self.train_dataset = self.dataset(self.config.data_cfg, SetType.train)
        self.validation_dataset = self.dataset(self.config.data_cfg, SetType.validation)

        # Train dataloader
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.train.batch_size, shuffle=True)

        # Evaluation train dataloader
        self.eval_train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.validation.batch_size)

        # Evaluation dataloader
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.config.batch_size, shuffle=True)

    # def _prepare_model(self):



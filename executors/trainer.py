import sys

import torch.optim.adamw

from utils.enums import SetType

from dataset.tinystories_dataset import TinyStoriesDataset
from torch.utils.data import DataLoader

from model.transformer import Transformer
from torch import nn
from utils.training_utils import cosine_annealing_with_warmup

from utils.logger import MLFlowLogger


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

    def _prepare_model(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Transformer(self.config).to(self.device)
        self.optimizer = torch.optim.adamw.AdamW(self.model.parameters(), lr=self.config.lr)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.config.data.preprocessing.special_tokens.index('[PAD]') - 1,
            label_smoothing=self.config.train.label_smoothing
        )
        lr_schedular = lambda step: cosine_annealing_with_warmup(
            cur_step=step, t_max=self.config.train.T_max, warmup_steps=self.config.train.warmup_steps
        )
        self.schedular = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedular)

    def _init_logger(self, init_logger):
        if init_logger:
            self.logger = MLFlowLogger(self.config.mlflow)
            if not self.config.train.continue_train:
                self.logger.log_hyperparameters(self.config)







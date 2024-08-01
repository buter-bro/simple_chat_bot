from executors.trainer import Trainer
from configs.experiment_config import experiment_cfg
import sys
from utils.enums import SetType
from torch.utils.data import DataLoader
from utils.data_utils import collate_function
from dataset.tinystories_dataset import TinyStoriesDataset
import torch
import os
import pandas as pd

import tracemalloc


def train():
    trainer = Trainer(experiment_cfg, init_logger=False)
    trainer.fit()
    # trainer.batch_overfit()

def predict(std_mode=False):
    trainer = Trainer(experiment_cfg, init_logger=False)
    dataset = getattr(sys.modules[__name__], experiment_cfg.data.name)

    if std_mode:
        model_path = 'experiments/no_f_ln_version/checkpoint_318657'
        input_text = ''
        while input_text != 'exit':
            input_text = input('Start a story: ')
            with open(os.path.join(experiment_cfg.data.path_to_data, 'test.txt'), 'w') as f:
                f.write(input_text)
            test_dataset = dataset(experiment_cfg.data, SetType.test)
            test_dataloader = DataLoader(
                test_dataset, experiment_cfg.validation.batch_size, collate_fn=collate_function, shuffle=False
            )
            predictions, sample_ids = trainer.predict(model_path, test_dataloader, experiment_cfg.inference)
            print(predictions)
    else:

        # Get data to make predictions on
        test_dataset = dataset(experiment_cfg.data, SetType.test)
        test_dataloader = DataLoader(
            test_dataset, experiment_cfg.train.validation_batch_size, collate_fn=collate_function, shuffle=False
        )

        # Get predictions
        model_path = experiment_cfg.best_checkpoint_name
        predictions, sample_ids = trainer.predict(model_path, test_dataloader, experiment_cfg.inference)

        # Save results to submission file
        test_results_df = pd.DataFrame({'ID': sample_ids, 'prediction': predictions})
        test_results_df['prediction'] = test_results_df['prediction'].replace('', ' ')
        test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    train()
    # predict(std_mode=True)




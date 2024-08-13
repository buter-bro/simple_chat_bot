from executors.trainer import Trainer
from configs.experiment_config import experiment_cfg
import sys
from utils.enums import SetType, InferenceMode
from torch.utils.data import DataLoader
from utils.data_utils import collate_function
from dataset.tinystories_dataset import TinyStoriesDataset
import torch
import os
import pandas as pd
from model.generate import Generate

import tracemalloc


def train():
    trainer = Trainer(experiment_cfg)
    trainer.fit()
    # trainer.batch_overfit()


def validate():
    trainer = Trainer(experiment_cfg, init_logger=False)

    validation_dataset = TinyStoriesDataset(experiment_cfg.data, SetType.validation)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=experiment_cfg.validation.batch_size,
        collate_fn=collate_function
    )
    model_path = 'experiments/no_f_ln_version/checkpoint_360914'
    trainer.load(model_path)
    total_loss, perplexity = trainer.evaluate(validation_dataloader)
    print(f'Loss: {total_loss}\nPerplexity: {perplexity}')


def predict(std_mode=False, inference_mode=InferenceMode.sentence):

    model_path = 'experiments/no_f_ln_version/checkpoint_360914'
    token_generator = Generate(experiment_cfg)

    if std_mode:
        while True:
            input_text = input('Start a story: ')
            if input_text == 'exit':
                break
            if input_text == '':
                input_text += '[SOS]'

            if inference_mode == InferenceMode.sentence:
                output_text = token_generator.generate_sequence(input_text)
                print(output_text)
            elif inference_mode == InferenceMode.token:
                for output_token in token_generator.token_generator(input_text):
                    print(output_token + ' ', end='')
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
    # train()
    predict(std_mode=True, inference_mode=InferenceMode.token)
    # validate()




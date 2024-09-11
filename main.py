from executors.trainer import Trainer
from configs.experiment_config import experiment_cfg
from configs.generation_config import generation_cfg
import sys
from utils.enums import SetType, InferenceMode
from torch.utils.data import DataLoader
from utils.data_utils import collate_function
from dataset.tinystories_dataset import TinyStoriesDataset
import torch
import os
import pandas as pd
from model.generate import Generate


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
    model_path = 'experiments/no_f_ln_version/best_checkpoint'
    trainer.load(model_path)
    total_loss, perplexity = trainer.evaluate(validation_dataloader)
    print(f'Loss: {total_loss}\nPerplexity: {perplexity}')


def predict(inference_mode=InferenceMode.sentence):

    token_generator = Generate(generation_cfg)

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
            for output_token in token_generator.token_generator_not_async(input_text):
                if output_token == '[EOS]':
                    break
                print(output_token + ' ', end='')


if __name__ == '__main__':
    train()
    # predict(inference_mode=InferenceMode.sentence)
    # validate()




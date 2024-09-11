import sys
import os
import random
import numpy as np
from tqdm.auto import tqdm
import torch

from torch.optim import AdamW

from utils.enums import SetType, InferenceType, InferenceMode
from typing import List

from dataset.tinystories_dataset import TinyStoriesDataset
from torch.utils.data import DataLoader
from utils.data_utils import collate_function

from model.transformer import Transformer, TransformerV1
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from utils.training_utils import cosine_annealing_with_warmup
from utils.data_utils import get_decoder_mask

from utils.logger import MLFlowLogger

from torcheval.metrics.text import Perplexity
from model.modules import RotaryPositionalEncoding
from torch.nn.functional import softmax


class Trainer:
    def __init__(self, config, init_logger=True):
        self.config = config

        self._init_logger(init_logger)
        self._prepare_data()
        self._prepare_model()

    def _prepare_data(self):
        self.dataset = getattr(sys.modules[__name__], self.config.data.name)

        self.train_dataset = self.dataset(self.config.data, SetType.train)
        self.validation_dataset = self.dataset(self.config.data, SetType.validation)

        # Train dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            collate_fn=collate_function,
        )

        # Evaluation train dataloader
        self.eval_train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.validation.batch_size,
            collate_fn=collate_function,
        )

        # Evaluation dataloader
        self.validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self.config.validation.batch_size,
            collate_fn=collate_function,
        )

    def _prepare_model(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        decoder_vocabulary_size = self.train_dataset.get_vocabulary_size()
        self.model = getattr(sys.modules[__name__], self.config.model_name)(
            self.config, decoder_vocabulary_size
        ).to(self.device)
        self.optimizer = getattr(sys.modules[__name__], self.config.optimizer)(
            self.model.parameters(), lr=self.config.train.lr
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.config.data.preprocessing.special_tokens.index('[PAD]') - 1,
            label_smoothing=self.config.train.label_smoothing
        )

        self.warmup_scheduler = LinearLR(
            optimizer=self.optimizer,
            start_factor=1. / self.config.train.warmup_steps,
            end_factor=1.,
            total_iters=self.config.train.warmup_steps
        )
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=len(self.train_dataloader) // self.config.train.accum_gradient_iter * self.config.train.num_epoches \
                  - self.config.train.warmup_steps
        )
        self.scheduler = SequentialLR(
            optimizer=self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[self.config.train.warmup_steps]
        )

        self.positional_encoding = RotaryPositionalEncoding(self.config.model.d_model, self.device)

        self.metric = Perplexity(ignore_index=self.config.data.preprocessing.special_tokens.index('[PAD]') - 1)
        self.metric.to(self.device)

    def _init_logger(self, init_logger):
        if init_logger:
            self.logger = MLFlowLogger(self.config.mlflow)
            if not self.config.train.continue_train:
                self.logger.log_hyperparameters(self.config)

    def save(self, filepath: str):
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            os.path.join(self.config.checkpoints_dir, filepath)
        )

    def get_last_checkpoint(self):
        checkpoints_list = [os.path.join(self.config.checkpoints_dir, f) for f in os.listdir(self.config.checkpoints_dir)]
        checkpoints_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints_list[0]

    def load(self, filepath: str):
        # checkpoint = torch.load(os.path.join(self.config.checkpoints_dir, filepath), map_location=self.device)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def update_params(self, valid_metric, best_metric):
        if best_metric < valid_metric:
            self.save(self.config.best_checkpoint_name)
            best_metric = valid_metric
        return best_metric

    def make_step(self, batch: torch.Tensor, accum_grad_finished: bool, update_model=True):

        _, decoder_inputs, decoder_outputs, decoder_mask = batch
        decoder_inputs = decoder_inputs.to(self.device)
        decoder_outputs = decoder_outputs.to(self.device)
        decoder_mask = decoder_mask.to(self.device)

        outputs = self.model(decoder_inputs, self.positional_encoding, decoder_mask)
        loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), decoder_outputs.reshape(-1) - 1)

        if update_model:
            self.optimizer.zero_grad()
            accum_loss = loss / self.config.train.accum_gradient_iter
            accum_loss.backward()
            if accum_grad_finished:
                self.optimizer.step()
                self.scheduler.step()

        return loss.item(), outputs.detach().cpu().numpy(), decoder_outputs.detach().cpu().numpy()

    def train_epoch(self, epoch):

        self.model.train()

        preprocessor = self.train_dataset.preprocessor

        train_losses, train_predictions, train_decoder_outputs = [], [], []
        pad_idx = self.config.data.preprocessing.special_tokens.index("[PAD]")

        steps_done = epoch * len(self.train_dataloader)
        steps_in_epoch = len(self.train_dataloader)

        for step, batch in tqdm(enumerate(self.train_dataloader), total=steps_in_epoch):
            if self.config.train.continue_train and step + steps_done <= self.config.train.checkpoint_from_step:
                continue

            # Gradient accumulation for memory efficiency
            accum_grad_finished = step % self.config.train.accum_gradient_iter == 0 or step == steps_in_epoch
            loss, output, decoder_outputs = self.make_step(batch, accum_grad_finished)
            if not accum_grad_finished:
                continue

            self.logger.save_metrics(
                SetType.train.name, 'learning_rate', self.optimizer.param_groups[0]['lr'], steps_done + step
            )

            # Adding 1 to token indexes to get predictions with PAD tokens
            prediction_with_pad = output.argmax(axis=-1) + 1
            train_losses.append(loss)
            train_predictions.extend(
                [prediction_with_pad[i][decoder_outputs[i] != pad_idx].tolist() for i in range(len(decoder_outputs))]
            )
            train_decoder_outputs.extend(decoder_outputs.tolist())

            # Evaluate performance on the validation data
            if step % self.config.train.validation_frequency == 0:
                print('Train evaluation started')
                val_interval = [0, self.config.train.validation_interval]
                valid_loss, valid_metric = self.evaluate(self.validation_dataloader, val_interval)
                print('Train evaluation finished')

                self.logger.save_metrics(SetType.validation.name, 'loss', valid_loss, step=steps_done + step)
                self.logger.save_metrics(SetType.validation.name, 'perplexity', valid_metric, step=steps_done + step)

            # Evaluate performance on the part of training data
            if step % self.config.train.log_frequency == 0 and step != 0:
                train_loss, train_metric, output_to_show = self.evaluate_train(
                    train_losses, train_predictions, train_decoder_outputs, preprocessor
                )

                self.logger.save_metrics(SetType.train.name, 'loss', train_loss, step=steps_done + step)
                self.logger.save_metrics(SetType.train.name, 'perplexity', train_metric, step=steps_done + step)
                print(output_to_show)
                train_losses, train_predictions, train_decoder_outputs = [], [], []
                torch.cuda.empty_cache()

            if step % self.config.checkpoint_save_frequency == 0:
                self.save(
                    self.config.checkpoint_name % (steps_done + step)
                )

                checkpoints_list = [os.path.join(self.config.checkpoints_dir, f) for f in
                                    os.listdir(self.config.checkpoints_dir)]
                checkpoints_list.sort(key=lambda x: os.path.getmtime(x))
                checkpoints_to_delete = checkpoints_list[:-self.config.checkpoint_files_count]
                for c in checkpoints_to_delete:
                    os.remove(os.path.join(self.config.checkpoints_dir, c))


    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, interval: List[int] = None, inference: bool = False):

        self.model.eval()
        pad_idx = self.config.data.preprocessing.special_tokens.index("[PAD]")
        total_loss, all_predictions, all_decoder_outputs = [], [], []

        dataloader_len = len(dataloader)

        for step, batch in enumerate(dataloader):

            if interval is not None and interval[0] < step:
                continue

            accum_grad_finished = step % self.config.train.accum_gradient_iter == 0 or step == dataloader_len
            loss, output, decoder_outputs = self.make_step(batch, accum_grad_finished, update_model=False)
            if not accum_grad_finished:
                continue

            total_loss.append(loss)

            self.metric.update(torch.tensor(output), torch.tensor(decoder_outputs) - 1)

            prediction_with_pad = output.argmax(axis=-1) + 1
            all_predictions.extend(
                [prediction_with_pad[i][decoder_outputs[i] != pad_idx].tolist() for i in
                 range(len(decoder_outputs))]
            )
            all_decoder_outputs.extend(decoder_outputs.tolist())

            if interval is not None and step > interval[1]:
                break

        total_loss = np.mean(total_loss)
        perplexity = self.metric.compute()
        self.metric.reset()

        if hasattr(torch.cuda, 'empty_cache') and interval is None:
            torch.cuda.empty_cache()

        all_predictions, all_decoder_outputs = [], []

        self.model.train()
        return total_loss, perplexity.item()

    def evaluate_train(self, losses: list[float], predictions: list[list[int]], decoder_outputs: list[list[int]],
                       preprocessor):

        losses = losses[-self.config.train.log_window:]
        predictions = predictions[-self.config.train.log_window:]
        decoder_outputs = decoder_outputs[-self.config.train.log_window:]

        train_predictions_decoded = preprocessor.decode(predictions, batch=True)
        train_targets_decoded = preprocessor.decode(decoder_outputs, batch=True)

        # self.metric.update(predictions, decoder_outputs)
        # perplexity = self.metric.compute()
        # self.metric.reset()

        random_sample_num = random.randint(0, len(predictions) - 1)
        output_to_show = f'Target:     {train_targets_decoded[random_sample_num]}\n' \
                         f'Prediction: {train_predictions_decoded[random_sample_num]}\n'

        return np.mean(losses), np.mean(np.exp(losses)), output_to_show

    @torch.no_grad()
    def inference(self, sequence: torch.Tensor, inference_config):
        self.model.eval()
        batch_size = sequence.size(0)
        input_size = sequence.size(-1)
        sos_token_id = self.config.data.preprocessing.special_tokens.index("[SOS]")
        eos_token_id = self.config.data.preprocessing.special_tokens.index("[EOS]")

        decoded_sequence = sequence

        inference_step = input_size - 1
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        while not finished_sequences.all() and inference_step < input_size + inference_config.stop_predict:

            decoder_mask = get_decoder_mask(decoded_sequence, device=self.device)
            output = self.model(decoded_sequence, self.positional_encoding, decoder_mask)

            if inference_config.type == InferenceType.greedy.value:
                current_token = torch.argmax(output, dim=-1)[:, -1].view(-1, 1) + 1
            elif inference_config.type == InferenceType.temperature.value:
                output = output / (inference_config.temperature_value + inference_config.eps)
                probabilities = softmax(output, dim=-1)
                current_token = probabilities[:, -1, :].multinomial(num_samples=1) + 1
            else:
                raise Exception('Unknown inference type!')

            decoded_sequence = torch.hstack([decoded_sequence, current_token])
            finished_sequences |= current_token.squeeze() == eos_token_id
            inference_step += 1

        eos_subsequence_mask = torch.cummax(decoded_sequence == eos_token_id, dim=1).values
        decoded_sequence = decoded_sequence.masked_fill(eos_subsequence_mask, eos_token_id)

        return decoded_sequence.cpu().tolist()

    @torch.no_grad()
    def predict(self, model_path: str, dataloader: DataLoader, inference_config):

        self.load(model_path)
        self.model.eval()

        preprocessor = self.train_dataset.preprocessor
        all_predictions, all_sample_ids = [], []

        for sample in dataloader:
            sample_id, decoder_inputs, decoder_outputs, decoder_mask = sample
            decoder_inputs = decoder_inputs.to(self.device)
            prediction = self.inference(decoder_inputs, inference_config)

            all_predictions.extend(preprocessor.decode(prediction, batch=True))
            all_sample_ids.extend(sample_id.view(-1).cpu().tolist())

        return all_predictions, all_sample_ids

    def fit(self):
        start_epoch, best_metric = 0, 0

        if self.config.train.continue_train:
            step = self.config.train.checkpoint_from_step
            last_checkpoint = self.get_last_checkpoint()
            self.load(last_checkpoint)
            start_epoch = step // len(self.train_dataloader)

        for epoch in range(start_epoch, self.config.train.num_epoches):
            print(f'Train epoch {epoch} started')
            self.train_epoch(epoch)
            print(f'Train epoch {epoch} finished')
            print(f'Validation epoch {epoch} started')
            _, valid_metric = self.evaluate(self.validation_dataloader)
            print(f'Validation epoch {epoch} finished')
            # _, eval_train_metric = self.evaluate(self.eval_train_dataloader)

            self.update_params(valid_metric, best_metric)

            step = (epoch + 1) * len(self.train_dataloader) - 1
            self.logger.save_metrics(SetType.validation.name + '_eval', 'perplexity', valid_metric, step=step)
            # self.logger.save_metrics(SetType.train.name + '_eval', 'perplexity', eval_train_metric, step=step)
            self.save(self.config.checkpoint_name % step)

    def batch_overfit(self):

        self.model.train()
        preprocessor = self.train_dataset.preprocessor
        pad_idx = self.config.data.preprocessing.special_tokens.index("[PAD]")
        batch = next(iter(self.train_dataloader))

        for step in range(self.config.overfit.num_iterations):

            accum_grad_finished = step % self.config.train.accum_gradient_iter == 0 or step == self.config.overfit.num_iterations
            loss, output, decoder_outputs = self.make_step(batch, accum_grad_finished)
            if not accum_grad_finished:
                continue

            if step % 10 == 0:
                prediction_with_pad = output.argmax(axis=-1)
                predictions = [
                    prediction_with_pad[i][decoder_outputs[i] != pad_idx].tolist() for i in range(len(decoder_outputs))
                ]

                perplexity = np.exp(loss)

                self.logger.save_metrics('overfit', 'perplexity', perplexity, step=step)
                self.logger.save_metrics('overfit', 'loss', loss, step=step)

                random_sample_num = random.randint(0, len(batch) - 1)

                print(f'Step: {step}')
                targets_decoded = preprocessor.tokenizer.decode(decoder_outputs[random_sample_num].tolist())
                predictions_decoded = preprocessor.tokenizer.decode((np.array(predictions[random_sample_num]) + 1).tolist())
                output_to_show = f'Target:     {targets_decoded}\n' \
                                 f'Prediction: {predictions_decoded}\n'
                print(output_to_show)






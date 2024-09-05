# noinspection PyUnresolvedReferences
from utils.tokenizer_utils.encode_fast import encode
# noinspection PyUnresolvedReferences
from utils.tokenizer_utils.decode_fast import decode

import torch
from utils.enums import InferenceMode, InferenceType
from utils.data_utils import get_decoder_mask
from model.modules import RotaryPositionalEncoding
from model.transformer import Transformer
from torch.nn.functional import softmax
from configs.data_config import data_cfg
from utils.common_functions import read_file
import os


class Generate:
    def __init__(self, config):
        self.config = config
        self.sos_token_id = self.config.data.preprocessing.special_tokens.index("[SOS]")
        self.eos_token_id = self.config.data.preprocessing.special_tokens.index("[EOS]")

        self._init_model(config.checkpoint_to_load)

        tokenizer_path = config.data.preprocessing.tokenizer_path
        if os.path.isfile(tokenizer_path):
            self.tokenizer = read_file(tokenizer_path)
        else:
            raise Exception("Tokenizer doesn't exist!")

        self.encode = encode
        self.decode = decode

    def _init_model(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        decoder_vocabulary_size = checkpoint['model_state_dict']['output.output_feed_forward.weight'].shape[0] + 1
        self.model = Transformer(self.config, decoder_vocabulary_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.positional_encoding = RotaryPositionalEncoding(self.config.model.d_model, self.device)

    @staticmethod
    def inference_output(output, temperature, eps):
        output = output / (temperature + eps)
        probabilities = softmax(output, dim=-1)
        return probabilities[:, -1, :].multinomial(num_samples=1) + 1

    def generate_token(self, input_text, inference_prefs=None):

        if inference_prefs is None:
            inference_prefs = {
                'temperature': self.config.inference.temperature_value,
                'stop_predict': self.config.inference.stop_predict,
                'eps': self.config.inference.eps
            }

        sequence = torch.tensor(self.encode(
            input_text,
            self.config.data.preprocessing.end_of_word,
            self.tokenizer['token2id']
        )[:-1]).unsqueeze(0).to(self.device)

        decoder_mask = get_decoder_mask(sequence, device=self.device)

        output = self.model(sequence, self.positional_encoding, decoder_mask)
        current_token = self.inference_output(output, **inference_prefs)

        output_token = self.decode(
            current_token.reshape(1).cpu().tolist(),
            self.tokenizer['id2token'],
            self.config.data.preprocessing.end_of_word,
            self.config.data.preprocessing.special_tokens,
            skip_special_tokens=False
        )
        return output_token

    async def token_generator(self, input_text, stop_predict=None, inference_prefs=None):
        if stop_predict is None:
            stop_predict = self.config.inference.stop_predict
        for i in range(stop_predict):
            output_token = self.generate_token(input_text, inference_prefs)
            input_text += output_token + ' '
            yield output_token

    def token_generator_not_async(self, input_text, stop_predict=None, inference_prefs=None):
        if stop_predict is None:
            stop_predict = self.config.inference.stop_predict
        for i in range(stop_predict):
            output_token = self.generate_token(input_text, inference_prefs)
            input_text += output_token + ' '
            yield output_token

    def generate_sequence(self, input_text, stop_predict=None, inference_prefs=None):

        if stop_predict is None:
            stop_predict = self.config.inference.stop_predict

        if inference_prefs is None:
            inference_prefs = {
                'temperature': self.config.inference.temperature_value,
                'eps': self.config.inference.eps
            }

        sequence = torch.tensor(self.encode(
            input_text,
            self.config.data.preprocessing.end_of_word,
            self.tokenizer['token2id']
        )[:-1]).to(self.device)

        decoded_sequence = sequence.unsqueeze(0)
        input_size = sequence.size(-1)
        inference_step = input_size - 1

        finished_sequences = torch.zeros(1, dtype=torch.bool, device=self.device)
        while not finished_sequences.all() and inference_step < input_size + stop_predict:
            decoder_mask = get_decoder_mask(decoded_sequence, device=self.device)
            output = self.model(decoded_sequence, self.positional_encoding, decoder_mask)
            current_token = self.inference_output(output, **inference_prefs)

            decoded_sequence = torch.hstack([decoded_sequence, current_token])
            finished_sequences |= current_token.squeeze() == self.eos_token_id
            inference_step += 1

        eos_subsequence_mask = torch.cummax(decoded_sequence == self.eos_token_id, dim=1).values
        decoded_sequence = decoded_sequence.masked_fill(eos_subsequence_mask, self.eos_token_id)

        output_text = self.decode(
            decoded_sequence.squeeze().cpu().tolist(),
            self.tokenizer['id2token'],
            self.config.data.preprocessing.end_of_word,
            self.config.data.preprocessing.special_tokens
        )

        return output_text








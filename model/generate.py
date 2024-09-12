import sys

# noinspection PyUnresolvedReferences
from utils.tokenizer_utils.encode_fast import encode as cython_encode
# noinspection PyUnresolvedReferences
from utils.tokenizer_utils.decode_fast import decode as cython_decode

import torch
from utils.enums import InferenceMode, InferenceType
from utils.data_utils import get_decoder_mask
from model.modules import RotaryPositionalEncoding
from model.transformer import Transformer, TransformerV1
from torch.nn.functional import softmax
from configs.data_config import data_cfg
from utils.common_functions import read_file
import os

import youtokentome as yttm


class Generate:
    def __init__(self, config):
        self.config = config
        self.sos_token_id = self.config.data.preprocessing.special_tokens.index("[SOS]")
        self.eos_token_id = self.config.data.preprocessing.special_tokens.index("[EOS]")

        self._init_model(config.checkpoint_to_load)

        self._init_tokenizer()

    def _init_tokenizer(self):
        tokenizer_path = self.config.data.preprocessing.tokenizer_path
        if not os.path.isfile(tokenizer_path):
            raise Exception("Tokenizer doesn't exist!")

        if self.config.tokenizer_name == 'YouTokenToMe':
            self.tokenizer = yttm.BPE(model=tokenizer_path)
            self.encode = self.tokenizer.encode
            self.decode = self.tokenizer.decode
        elif self.config.tokenizer_name == 'BPETokenizer':
            self.tokenizer = read_file(tokenizer_path)
            self.encode = cython_encode
            self.decode = cython_decode
        else:
            raise Exception("Tokenizer doesn't exist!")

    def _init_model(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        decoder_vocabulary_size = checkpoint['model_state_dict']['output.output_feed_forward.weight'].shape[0] + 1
        self.model = getattr(sys.modules[__name__], self.config.model_name)(self.config, decoder_vocabulary_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.positional_encoding = RotaryPositionalEncoding(self.config.model.d_model, self.device)

    @staticmethod
    def inference_output(output, temperature, eps):
        output = output / (temperature + eps)
        probabilities = softmax(output, dim=-1)
        return probabilities[:, -1, :].multinomial(num_samples=1) + 1

    def get_encode_prefs(self, tokenizer_name, input_text):
        if tokenizer_name == 'BPETokenizer':
            return {
                'text': input_text,
                'end_of_word': self.config.data.preprocessing.end_of_word,
                'token2id': self.tokenizer['token2id']
            }
        elif tokenizer_name == 'YouTokenToMe':
            return {
                'sentences': input_text,
                'output_type': yttm.OutputType.ID,
                'bos': True,
                'eos': True
            }

    def get_decode_prefs(self, tokenizer_name, token):
        if tokenizer_name == 'BPETokenizer':
            return {
                'tokens': token,
                'id2token': self.tokenizer['id2token'],
                'end_of_word': self.config.data.preprocessing.end_of_word,
                'special_tokens': self.config.data.preprocessing.special_tokens,
                'skip_special_tokens': False
            }
        elif tokenizer_name == 'YouTokenToMe':
            return {
                'ids': token
            }

    def generate_token(self, input_text, inference_prefs=None):

        if inference_prefs is None:
            inference_prefs = {
                'temperature': self.config.inference.temperature_value,
                'stop_predict': self.config.inference.stop_predict,
                'eps': self.config.inference.eps
            }

        encode_prefs = self.get_encode_prefs(self.config.tokenizer_name, input_text)

        sequence = torch.tensor(self.encode(**encode_prefs)[:-1]).unsqueeze(0).to(self.device)

        decoder_mask = get_decoder_mask(sequence, device=self.device)

        output = self.model(sequence, self.positional_encoding, decoder_mask)
        current_token = self.inference_output(output, **inference_prefs)

        decode_prefs = self.get_decode_prefs(self.config.tokenizer_name, current_token.reshape(1).cpu().tolist())

        output_token = self.decode(**decode_prefs)
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

        encode_prefs = self.get_encode_prefs(self.config.tokenizer_name, input_text)
        sequence = torch.tensor(self.encode(**encode_prefs)[:-1]).to(self.device)

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

        decode_prefs = self.get_decode_prefs(self.config.tokenizer_name, decoded_sequence.squeeze().cpu().tolist())

        output_text = self.decode(**decode_prefs)

        return output_text








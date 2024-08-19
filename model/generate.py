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

        tokenizer_path = os.path.join(config.data.path_to_data, config.data.preprocessing.tokenizer_path)
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

    def inference_output(self, output, inference_config):
        if inference_config.type == InferenceType.greedy.value:
            return torch.argmax(output, dim=-1)[:, -1].view(-1, 1) + 1
        elif inference_config.type == InferenceType.temperature.value:
            output = output / (inference_config.temperature_value + inference_config.eps)
            probabilities = softmax(output, dim=-1)
            return probabilities[:, -1, :].multinomial(num_samples=1) + 1
        else:
            raise Exception('Unknown inference type!')

    def generate_token(self, input_text):

        sequence = torch.tensor(self.encode(
            input_text,
            self.config.data.preprocessing.end_of_word,
            self.tokenizer['token2id']
        )[:-1]).unsqueeze(0).to(self.device)

        decoder_mask = get_decoder_mask(sequence, device=self.device)

        output = self.model(sequence, self.positional_encoding, decoder_mask)
        current_token = self.inference_output(output, self.config.inference)

        output_token = self.decode(
            current_token.reshape(1).cpu().tolist(),
            self.tokenizer['id2token'],
            self.config.data.preprocessing.end_of_word,
            self.config.data.preprocessing.special_tokens,
            skip_special_tokens=False
        )
        return output_token

    async def token_generator(self, input_text):
        for i in range(self.config.inference.stop_predict):
            output_token = self.generate_token(input_text)
            input_text += output_token + ' '
            yield output_token

    def generate_sequence(self, input_text):

        sequence = torch.tensor(self.encode(
            input_text,
            self.config.data.preprocessing.end_of_word,
            self.tokenizer['token2id']
        )[:-1]).to(self.device)

        decoded_sequence = sequence.unsqueeze(0)
        input_size = sequence.size(-1)
        inference_step = input_size - 1

        finished_sequences = torch.zeros(1, dtype=torch.bool, device=self.device)
        while not finished_sequences.all() and inference_step < input_size + self.config.inference.stop_predict:
            decoder_mask = get_decoder_mask(decoded_sequence, device=self.device)
            output = self.model(decoded_sequence, self.positional_encoding, decoder_mask)
            current_token = self.inference_output(output, self.config.inference)

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








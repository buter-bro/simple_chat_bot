from torch.utils.data import Dataset
from dataset.preprocessing import Preprocessing

import os
from utils.enums import SetType
from utils.common_functions import read_file, write_file


class TinyStoriesDataset(Dataset):
    def __init__(self, config, set_type: SetType):
        self.config = config
        self.set_type = set_type
        self._init_preprocessors()
        self._get_data()

    def _init_preprocessors(self):
        raw_data_path = os.path.join(
            self.config.path_to_data, self.config.preprocessing.raw_data_path_template % self.set_type.name
        )
        self.preprocessor = Preprocessing(self.config.preprocessing, raw_data_path, self.config.vocabulary_size)

    def _get_data(self):
        """Gets data to pass to Transformer model."""

        preprocessed_data_path = os.path.join(
            self.config.path_to_data,
            self.config.preprocessing.preprocessed_data_path_template % self.set_type.name
        )
        tokenizer_path_to_load = os.path.join(
            self.config.path_to_data, self.config.preprocessing.tokenizer_path
        )

        if not os.path.exists(preprocessed_data_path) or self.set_type is SetType.test:
            self.encode_data(tokenizer_path_to_load, preprocessed_data_path)
        else:
            self.dataset = read_file(preprocessed_data_path)
            self.preprocessor.load_tokenizer_state(tokenizer_path_to_load)

    def encode_data(self, tokenizer_path_to_load: str, preprocessed_data_path: str):
        """Encoding data using BPETokenizer"""
        self.preprocessor.train(tokenizer_path_to_load)

        raw_data_path = os.path.join(
            self.config.path_to_data, self.config.preprocessing.raw_data_path_template % self.set_type.name
        )
        raw_data = open(raw_data_path, encoding='utf-8').readlines()

        self.dataset = []

        for idx, text in enumerate(raw_data):
            tokens = self.preprocessor.encode(text)
            self.dataset.append(tokens)

        write_file(self.dataset, preprocessed_data_path)

    def get_vocabulary_size(self):
        return self.preprocessor.tokenizer.get_vocab_size()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample_data = {
            'sample_id': idx,
            'tokens': self.dataset[idx]
        }
        return sample_data


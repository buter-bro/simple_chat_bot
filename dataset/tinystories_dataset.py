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
        raw_data_path = os.path.join(self.config.path_to_data, f"{SetType.train.name}.csv")
        self.preprocessors = Preprocessing(self.config.preprocessing, raw_data_path, self.config.vocabulary_size)

    def _get_data(self):
        """Gets data to pass to Transformer model."""

        if self.set_type == SetType.test:
            self.dataset = [{'id': idx, 'tokens': []} for idx in range(len(self.dataset))]
            return

        preprocessed_data_path = os.path.join(
            self.config.path_to_data,
            self.config.preprocessing.preprocessed_data_path_template % self.set_type.name
        )
        tokenizer_path_to_load = os.path.join(
            self.config.path_to_data, self.config.preprocessing.tokenizer_path
        )

        if not os.path.exists(preprocessed_data_path):
            self.encode_data(tokenizer_path_to_load, preprocessed_data_path)
        else:
            self.dataset = read_file(preprocessed_data_path)
            self.preprocessors.load_tokenizer_state(tokenizer_path_to_load)

    def encode_data(self, tokenizer_path_to_load: str, preprocessed_data_path: str):

        self.preprocessors.train(tokenizer_path_to_load)

        raw_data_path = os.path.join(
            self.config.path_to_data, self.config.preprocessing.raw_data_path_template % self.set_type.name
        )
        raw_data = open(raw_data_path, encoding='utf-8').readlines()

        self.dataset = []

        for idx, text in enumerate(raw_data):
            tokens = self.preprocessors.encode(text)
            self.dataset.append({'id': idx, 'tokens': tokens})

        write_file(self.dataset, preprocessed_data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample_data = {
            'sample_pair_id': self.dataset[idx]['id'],
            'tokens': self.dataset[idx]['tokens']
        }
        return sample_data


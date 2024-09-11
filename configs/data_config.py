import os
from easydict import EasyDict

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_cfg = EasyDict()

data_cfg.tinystories_dataset = EasyDict()
data_cfg.tinystories_dataset.name = 'TinyStoriesDataset'
data_cfg.tinystories_dataset.path_to_data = f'{ROOT_DIR}/data/tinystories_dataset_v1'
data_cfg.tinystories_dataset.vocabulary_size = 20000

data_cfg.tinystories_dataset.preprocessing = EasyDict()
data_cfg.tinystories_dataset.preprocessing.tokenizer = 'YouTokenToMe'
data_cfg.tinystories_dataset.preprocessing.raw_data_path_template = '%s.txt'
data_cfg.tinystories_dataset.preprocessing.path_to_data = f'{ROOT_DIR}/data/tinystories_dataset_v1'
data_cfg.tinystories_dataset.preprocessing.tokenizer_path = f'{ROOT_DIR}/export/tokenization.pickle'
data_cfg.tinystories_dataset.preprocessing.preprocessed_data_path_template = 'tokenized_data_%s.pickle'  # set type
data_cfg.tinystories_dataset.preprocessing.special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
data_cfg.tinystories_dataset.preprocessing.lowercase = False
data_cfg.tinystories_dataset.preprocessing.end_of_word = '</w>'



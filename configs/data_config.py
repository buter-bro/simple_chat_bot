import os
from easydict import EasyDict

data_cfg = EasyDict()

data_cfg.tinystories_dataset = EasyDict()
data_cfg.tinystories_dataset.name = 'TinyStoriesDataset'
data_cfg.tinystories_dataset.path_to_data = '/kaggle/input/tinystories-dataset-v1/tinystories_dataset_v1'
data_cfg.tinystories_dataset.vocabulary_size = 15000

data_cfg.tinystories_dataset.preprocessing = EasyDict()
data_cfg.tinystories_dataset.preprocessing.tokenizer = 'BPETokenizer'
data_cfg.tinystories_dataset.preprocessing.raw_data_path_template = '%s.txt'
data_cfg.tinystories_dataset.preprocessing.tokenizer_path = 'tokenization.pickle'
data_cfg.tinystories_dataset.preprocessing.preprocessed_data_path_template = 'tokenized_data_%s.pickle'  # set type
data_cfg.tinystories_dataset.preprocessing.special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
data_cfg.tinystories_dataset.preprocessing.end_of_word = '</w>'



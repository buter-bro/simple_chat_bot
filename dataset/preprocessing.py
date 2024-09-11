import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Union
from tqdm import tqdm
import youtokentome as yttm


class BPETokenizer:
    """A class for Byte-Pair Encoding (BPE) tokenizer implementation.

    This tokenization algorithm originally proposed in
        'Neural Machine Translation of Rare Words with Subword Units' (https://aclanthology.org/P16-1162.pdf)
    """

    def __init__(self, config, vocabulary_size):
        """Tokenizer initialization.

        Args:
            special_tokens: special tokens list ([SOS], [EOS], [PAD] etc)
            end_of_word: a special character sequence representing the end of a word
            vocabulary_size: desired vocabulary size
            lowercase: if to lowercase training texts
        """
        self.config = config
        self.special_tokens = config.special_tokens
        self.end_of_word = config.end_of_word
        self.use_unk_token = '[UNK]' in self.special_tokens
        self.use_bounds_tokens = '[SOS]' in self.special_tokens and '[EOS]' in self.special_tokens
        self.vocabulary_size = vocabulary_size
        self.lowercase = config.lowercase

        self._init_vocabulary()

    def _init_vocabulary(self, id2token=None, token2id=None):
        """Initializes vocabulary."""
        if all((id2token, token2id)):
            self.id2token, self.token2id = id2token, token2id
        else:
            tokens_to_add = self.special_tokens + list(map(str, range(10)))

            self.id2token = {i: token for i, token in enumerate(tokens_to_add)}
            self.token2id = {token: i for i, token in enumerate(tokens_to_add)}

    def get_words(self, file_path: str) -> dict:
        """Gets a word counter based on the training text corpus.

        Args:
            file_path: a path to the text file where each raw represents one training sequence (sentence)

        Returns:
            A dictionary with keys represented as words (as tuples of characters) and values as word frequencies
        """
        word_counts = defaultdict(int)

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if self.lowercase:
                    line = line.lower().strip()
                else:
                    line = line.strip()
                line = re.sub(r'\d+', '', line)

                for word in line.split():
                    word_counts[word] += 1

        word_counts = dict([(tuple(x[:-1]) + (x[-1] + self.end_of_word,), y) for (x, y) in word_counts.items()])
        return word_counts

    @staticmethod
    def prepare_tokens(word_counts: dict) -> list:
        """Gets tokens from the words counter.

        Args:
            word_counts: A dictionary with keys represented as words (as tuples of tokens)
                    and values as word frequencies

        Returns:
            A sequence with tokens sorted in reverse order by length
        """
        tokens = set()
        for word in word_counts.keys():
            tokens.update(word)

        return sorted(tokens, key=lambda x: len(x), reverse=True)

    @staticmethod
    def get_pair_statistics(word_counts: dict) -> dict:
        """Gets pairs frequency statistics.

        Args:
            word_counts: A dictionary with keys represented as words (as tuples of tokens)
                    and values as word frequencies

        Returns:
            A dictionary with keys as a tuples of token pairs and values as these pairs frequency
        """
        pairs = defaultdict(int)
        for word, freq in word_counts.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def merge(word_counts: dict, most_frequent_pair: tuple) -> dict:
        """Makes tokens merging step.

        Args:
            word_counts: A dictionary with keys represented as words (as tuples of tokens)
                    and values as word frequencies
            most_frequent_pair: a pair of the most frequently occurring tokens

        Returns:
            A dictionary with keys represented as words (as merged tokens) and values as word frequencies
        """
        new_vocab = {}
        first, second = most_frequent_pair
        pair_str = ''.join(most_frequent_pair)
        pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')

        for word, freq in word_counts.items():
            new_word = ' '.join(word)
            new_word = pattern.sub(pair_str, new_word)
            new_word = tuple(new_word.split(' '))
            new_vocab[new_word] = freq

        return new_vocab

    def _update_vocabulary(self, word_counts: dict):
        """Updates vocabulary with new tokens.

        Args:
            word_counts: A dictionary with keys represented as words (as tuples of tokens)
                    and values as word frequencies
        """
        tokens = self.prepare_tokens(word_counts)
        shift = len(self.id2token)

        # Add tokens from merged vocabulary
        for i, token in enumerate(tokens):
            self.id2token[i + shift] = token
            self.token2id[token] = i + shift

        # Add all unique symbols
        unique_symbols = set(''.join(tokens))
        shift, added = len(self.id2token), 0
        for unique_symbol in unique_symbols:
            if unique_symbol in self.token2id:
                continue
            self.id2token[added + shift] = unique_symbol
            self.token2id[unique_symbol] = added + shift
            added += 1

        # Add end of word symbol
        self.id2token[added + shift] = self.end_of_word
        self.token2id[self.end_of_word] = added + shift

    def train(self, file_path: str):
        """Makes BPE training.

        Args:
            file_path: a path to the text file where each raw represents one training sequence (sentence)
        """
        words = self.get_words(file_path)

        for i in tqdm(range(self.vocabulary_size)):
            pairs = self.get_pair_statistics(words)
            most_frequent_pair = max(pairs, key=pairs.get)
            words = self.merge(words, most_frequent_pair)

        self._update_vocabulary(words)

    def encode(self, text: str, tokenize=False) -> Union[list[int], list[str]]:
        """Encodes text into token ids.

        Args:
            text: a text sequence to encode

        Returns:
            A list of tokens for input text
        """
        text = text.lower() if self.lowercase else text
        words = [tuple(word) + (self.end_of_word,) for word in text.strip().split()]
        if self.use_bounds_tokens:
            words = [('[SOS]',)] + words + [('[EOS]',)]
        encoded = []
        tokens = []

        for word in words:
            i = 0
            while i < len(word):
                unknown = True
                for j in range(len(word), i, -1):
                    subword = ''.join(word[i:j])
                    if subword in self.token2id:
                        encoded.append(self.token2id[subword])
                        tokens.append(subword)
                        i = j - 1
                        unknown = False
                        break
                i += 1
                if unknown and self.use_unk_token:
                    encoded.append(self.token2id['[UNK]'])
                    tokens.append('[UNK]')
        if tokenize:
            return tokens
        return encoded

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Decodes token ids into text.

        Args:
            tokens: a list of token ids
            skip_special_tokens: a boolean indicating whether to drop special tokens from decoded sequence

        Returns:
            Decoded text
        """
        decoded = ''.join([self.id2token[t] for t in tokens])
        if self.use_bounds_tokens and not skip_special_tokens:
            decoded = decoded.replace('[SOS]', f'[SOS]{self.end_of_word}').replace('[EOS]', f'[EOS]{self.end_of_word}')

        if skip_special_tokens:
            decoded = re.sub('|'.join(map(re.escape, self.special_tokens)), ' ', decoded).strip()
        return decoded.replace(self.end_of_word, ' ').strip()

    def tokenize(self, text: str) -> list[str]:
        return self.encode(text, tokenize=True)

    def save(self, path: str):
        """Saves vocabulary state."""
        with open(path, 'wb') as f:
            pickle.dump({'id2token': self.id2token, 'token2id': self.token2id}, f)

    def load(self, path):
        """Loads vocabulary state."""
        with open(path, 'rb') as f:
            vocab_mappings = pickle.load(f)

        self._init_vocabulary(vocab_mappings['id2token'], vocab_mappings['token2id'])

    def get_vocab_size(self) -> int:
        """Gets vocabulary size."""
        return len(self.id2token)


class YouTokenToMe:
    def __init__(self, config, vocabulary_size):
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.use_bounds_tokens = '[SOS]' in self.config.special_tokens and '[EOS]' in self.config.special_tokens

    def load(self, path):
        self.tokenizer = yttm.BPE(model=path)

    def train(self):
        self.tokenizer = yttm.BPE.train(
            data=os.path.join(self.config.path_to_data, 'train.txt'),
            model=self.config.tokenizer_path,
            vocab_size=self.vocabulary_size,
            coverage=1,
            n_threads=-1,
            pad_id=0,
            unk_id=3,
            bos_id=1,
            eos_id=2
        )

    def tokenize(self, text):
        return self.tokenizer.encode(text, output_type=yttm.OutputType.SUBWORD)

    def encode(self, text: str):
        if self.use_bounds_tokens:
            return self.tokenizer.encode(text, output_type=yttm.OutputType.ID, bos=True, eos=True)
        return self.tokenizer.encode(text, output_type=yttm.OutputType.ID)

    def decode(self, tokens: list, skip_special_tokens: bool = True):
        if skip_special_tokens:
            return self.tokenizer.decode(tokens, ignore_ids=[0, 1, 2, 3])
        return self.tokenizer.decode(tokens)

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

class Preprocessing:
    """A class for raw text data preprocessing using BPE algorithm."""

    def __init__(self, config, train_data_path: str, vocabulary_size: int):
        self.config = config
        self.train_data_path = train_data_path

        self._init_tokenizer(vocabulary_size)

    def _init_tokenizer(self, vocabulary_size: int):
        """Initializes BPE tokenizer."""
        self.tokenizer = getattr(sys.modules[__name__], self.config.tokenizer)(self.config, vocabulary_size)

    def train(self, tokenizer_path: str):
        """Gets trained BPE tokenizer."""
        if self.config.tokenizer == 'BPETokenizer':
            if os.path.exists(tokenizer_path):
                self.load_tokenizer_state(tokenizer_path)
            else:
                self.tokenizer.train(self.train_data_path)
                self.save_tokenizer_state(tokenizer_path)
        elif self.config.tokenizer == 'YouTokenToMe':
            if os.path.exists(tokenizer_path):
                self.load_tokenizer_state(tokenizer_path)
            else:
                self.tokenizer.train()
        else:
            raise Exception("Tokenizer doesn't exist!")

    def save_tokenizer_state(self, path_to_save: str):
        """Saves tokenizer state."""
        self.tokenizer.save(path_to_save)

    def load_tokenizer_state(self, path_to_load: str):
        """Loads tokenizer state."""
        self.tokenizer.load(path_to_load)

    def encode(self, text: Union[str, list[str]], batch=False) -> Union[list[int], list[list[int]]]:
        """Encodes text into token ids.

        Args:
            text: a text sequence to encode
            batch: if input text is a batch of text sequences

        Returns:
            A list with token ids or a batch of lists with token ids
        """
        if batch:
            return [self.tokenizer.encode(t) for t in text]
        return self.tokenizer.encode(text)

    def tokenize(self, text: Union[str, list[str]], batch=False) -> Union[list[str], list[list[str]]]:
        """Encodes text into token.

        Args:
            text: a text sequence to encode
            batch: if input text is a batch of text sequences

        Returns:
            A list with token or a batch of lists with token
        """
        if batch:
            return [self.tokenizer.tokenize(t) for t in text]
        return self.tokenizer.tokenize(text)

    def decode(self, tokens, batch=False, skip_special_tokens=True) -> Union[str, list[str]]:
        """Decodes token ids into text.

        Args:
            tokens: a list of token ids
            batch: if input tokens is a batch of sequence tokens
            skip_special_tokens: a boolean indicating whether to drop special tokens from decoded sequence

        Returns:
            A decoded text or a batch of lists with decoded texts
        """
        if batch:
            return [self.tokenizer.decode(sample, skip_special_tokens) for sample in tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens)

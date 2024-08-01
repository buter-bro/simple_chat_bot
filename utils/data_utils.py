import torch
from torch.nn.utils.rnn import pad_sequence


def get_decoder_mask(sequence: torch.Tensor, pad_id: int = 0, device='cpu'):

    batch_size, batch_max_seq_len = sequence.size()
    padding_mask = (sequence == pad_id).unsqueeze(1).unsqueeze(2)
    future_positions_mask = torch.triu(torch.ones((batch_max_seq_len, batch_max_seq_len), device=device), diagonal=1).bool()
    return torch.max(padding_mask, future_positions_mask)


def collate_function(batch):
    decoder_inputs, decoder_outputs, sample_indices = [], [], []

    for sample in batch:
        decoder_inputs.append(torch.tensor(sample['tokens'][:-1]))
        decoder_outputs.append(torch.tensor(sample['tokens'][1:]))
        sample_indices.append(torch.tensor(sample['sample_id']))

    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True)
    decoder_outputs = pad_sequence(decoder_outputs, batch_first=True)
    sample_indices = torch.vstack(sample_indices)

    decoder_mask = get_decoder_mask(decoder_inputs)

    return sample_indices, decoder_inputs, decoder_outputs, decoder_mask


def preprocess_data_files(raw_file_path: str, preprocessed_file_path: str):
    stories = []
    with open(raw_file_path, 'r', encoding="utf8") as file:
        story = ''
        for line in file.read().splitlines():
            if line == '<|endoftext|>':
                stories.append(story)
                story = ''
                continue
            story += line
    with open(preprocessed_file_path, 'w', encoding="utf8") as file:
        for line in stories:
            file.write(line + '\n')



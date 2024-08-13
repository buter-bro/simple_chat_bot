import re 

def decode(tokens: list[int], id2token, end_of_word, special_tokens, use_bounds_tokens=False, skip_special_tokens: bool = True) -> str:
    """Decodes token ids into text.

    Args:
        tokens: a list of token ids
        skip_special_tokens: a boolean indicating whether to drop special tokens from decoded sequence

    Returns:
        Decoded text
    """
    decoded = ''.join([id2token[t] for t in tokens])
    if use_bounds_tokens and not skip_special_tokens:
        decoded = decoded.replace('[SOS]', f'[SOS]{end_of_word}').replace('[EOS]', f'[EOS]{end_of_word}')

    if skip_special_tokens:
        decoded = re.sub('|'.join(map(re.escape, special_tokens)), ' ', decoded).strip()
    return decoded.replace(end_of_word, ' ').strip()
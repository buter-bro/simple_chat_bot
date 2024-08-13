
def encode(text: str, end_of_word: str, token2id, tokenize=False, lowercase=False, use_bounds_tokens=True, use_unk_token=True):
    """Encodes text into token ids.

    Args:
        text: a text sequence to encode

    Returns:
        A list of tokens for input text
    """
    text = text.lower() if lowercase else text
    words = [tuple(word) + (end_of_word,) for word in text.strip().split()]
    if use_bounds_tokens:
        words = [('[SOS]',)] + words + [('[EOS]',)]
    encoded = []
    tokens = []

    for word in words:
        i = 0
        while i < len(word):
            unknown = True
            for j in range(len(word), i, -1):
                subword = ''.join(word[i:j])
                if subword in token2id:
                    encoded.append(token2id[subword])
                    tokens.append(subword)
                    i = j - 1
                    unknown = False
                    break
            i += 1
            if unknown and use_unk_token:
                encoded.append(token2id['[UNK]'])
                tokens.append('[UNK]')
    if tokenize:
        return tokens
    return encoded
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer


def compression_curve(tokenizer: BPETokenizer) -> list[int]:
    """
    Returns a list showing the total number of tokens as a function of the number of merge rules applied.
    """
    tokens = tokenizer.tokens
    num_tokens = sum(
        tokens[token_id].original_count for token_id in tokens.keys() if token_id in tokenizer.pretokenizer.atomic_tokens
    )
    curve = [num_tokens]

    for merge_rule in tokenizer.merge_rules:
        num_tokens -= tokens[merge_rule.token_to].original_count
        curve.append(num_tokens)

    return curve

from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer


def compression_curve(tokenizer: BPETokenizer) -> list[int]:
    """
    Returns a list showing the total number of tokens as a function of the number of merge rules applied.
    """
    token_data = {t["id"]: t for t in tokenizer.metadata["tokens"]}
    num_tokens = sum(t["original_count"] for id, t in token_data.items() if id in tokenizer.pretokenizer.atomic_tokens)
    curve = [num_tokens]

    for merge_rule in tokenizer.merge_rules:
        num_tokens -= token_data[merge_rule.token_to]["original_count"]
        curve.append(num_tokens)

    return curve

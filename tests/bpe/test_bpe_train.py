from collections import Counter

import pytest

from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.bpe.tokenizer import MergeRule
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import PRETOKENIZER_REGISTRY, get_pretokenizer
from script_bpe.utils import token_array
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig


def taylor_swift_text():
    with open("tests/data/taylorswift.txt", "r") as f:
        return f.read()


def zeros_text():
    return "\n".join([(" " + "0" * i) * j for i in range(10) for j in range(5)])


EXPECTED_MERGE_RULES = {
    ("taylor_swift_text", "scriptenc"): [
        MergeRule(tokens_from=(1537, 3), token_to=1917),
        MergeRule(tokens_from=(1917, 1550), token_to=1918),
        MergeRule(tokens_from=(1550, 31), token_to=1919),
        MergeRule(tokens_from=(1919, 1550), token_to=1920),
        MergeRule(tokens_from=(1550, 35), token_to=1921),
        MergeRule(tokens_from=(41, 1550), token_to=1922),
        MergeRule(tokens_from=(1921, 1550), token_to=1923),
        MergeRule(tokens_from=(27, 1550), token_to=1924),
        MergeRule(tokens_from=(1550, 1922), token_to=1925),
        MergeRule(tokens_from=(1550, 1924), token_to=1926),
        MergeRule(tokens_from=(1531, 3), token_to=1927),
        MergeRule(tokens_from=(1550, 46), token_to=1928),
        MergeRule(tokens_from=(1550, 44), token_to=1929),
        MergeRule(tokens_from=(1550, 34), token_to=1930),
        MergeRule(tokens_from=(1532, 14), token_to=1931),
        MergeRule(tokens_from=(1927, 1531), token_to=1932),
        MergeRule(tokens_from=(1550, 45), token_to=1933),
        MergeRule(tokens_from=(1532, 12), token_to=1934),
        MergeRule(tokens_from=(1550, 38), token_to=1935),
        MergeRule(tokens_from=(40, 1550), token_to=1936),
    ],
    ("zeros_text", "bytes_gpt4"): [
        MergeRule(tokens_from=(49, 49), token_to=257),
        MergeRule(tokens_from=(257, 49), token_to=258),
        MergeRule(tokens_from=(11, 11), token_to=259),
        MergeRule(tokens_from=(33, 33), token_to=260),
        MergeRule(tokens_from=(11, 260), token_to=261),
        MergeRule(tokens_from=(33, 261), token_to=262),
        MergeRule(tokens_from=(11, 262), token_to=263),
        MergeRule(tokens_from=(260, 259), token_to=264),
        MergeRule(tokens_from=(261, 262), token_to=265),
        MergeRule(tokens_from=(263, 265), token_to=266),
        MergeRule(tokens_from=(266, 264), token_to=267),
    ],
}


# Linearized parametrization with pytest.param
@pytest.mark.parametrize(
    "pretokenizer_name, text_fixture, expected_merge_rules",
    [
        # Most cases: expected_merge_rules=None
        *[
            pytest.param(
                pretokenizer_name,
                text_fixture,
                EXPECTED_MERGE_RULES.get((text_fixture.__name__, pretokenizer_name), None),
                id=f"{pretokenizer_name}-{text_fixture.__name__}",
            )
            for pretokenizer_name in PRETOKENIZER_REGISTRY
            if "nosplit" not in pretokenizer_name
            for text_fixture in [taylor_swift_text, zeros_text]
        ]
    ],
)

def test_bpe_train(tmp_path, pretokenizer_name, text_fixture, expected_merge_rules, x_tokens=20):
    text = text_fixture()
    pretokenizer = get_pretokenizer(pretokenizer_name)
    corpus = PretokenizedCorpus.from_texts(
        f"test_bpe_train_{text_fixture}", texts=[text], pretokenizer=pretokenizer, base_path=str(tmp_path)
    )
    trainer = BPETrainer(pretokenizer, corpus, BPETrainerConfig(additional_vocab_size=x_tokens, verbose=True))
    tokenizer = trainer.train()

    # Basic assertions
    assert isinstance(tokenizer, BPETokenizer)
    if "gpt4" not in pretokenizer_name:
        assert len(tokenizer.merge_rules) == x_tokens, (
            f"Expected {x_tokens} merge rules, got {len(tokenizer.merge_rules)}"
        )
    assert len(tokenizer.tokens) == len(tokenizer.merge_rules) + len(pretokenizer.atomic_tokens), (
        f"Expected {len(tokenizer.tokens)}"
    )

    # Verify token counts
    metadata_tokens = {t["id"]: t for t in tokenizer.metadata["tokens"]}
    tokens = tokenizer.encode(text)
    atomic_tokens = sum(tokenizer.pretokenizer.pretokenize(text), token_array([]))

    for token_id, count in Counter(atomic_tokens).items():
        assert count == metadata_tokens[token_id]["original_count"], (
            f"Base token {token_id} original count mismatch: manual count {count} does not match metadata {metadata_tokens[token_id]}"
        )

    for token_id, count in Counter(tokens).items():
        assert count == metadata_tokens[token_id]["final_count"], (
            f"Token {token_id} final count mismatch: manual count {count} does not match metadata {metadata_tokens[token_id]}"
        )

    # Optional merge rules check
    if expected_merge_rules is not None:
        assert tokenizer.merge_rules == expected_merge_rules, (
            f"Merge rules mismatch for {pretokenizer_name} + {text_fixture.__name__}"
        )

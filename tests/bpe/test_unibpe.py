import math
from collections import Counter

import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.utils import token_array

# "xy" is rarer than "pq" but x and y occur nowhere else, while p and q are frequent on their own. Count-ranked
# BPE merges (p, q) first; UniBPE merges (x, y) because it removes two whole tokens from the Unigram model.
CRAFTED_TEXT = "xy\n" * 5 + "pq\n" * 6 + "p\n" * 30 + "q\n" * 30


def h(x: float) -> float:
    return x * math.log(x) if x > 0 else 0.0


@pytest.fixture
def pretokenizer():
    return get_pretokenizer("bytes_gpt4")


def make_corpus(tmp_path, pretokenizer, text: str, name: str):
    return PretokenizedCorpus.from_texts(name, texts=[text], pretokenizer=pretokenizer, base_path=str(tmp_path))


def train(tmp_path, pretokenizer, text: str, name: str, **config):
    corpus = make_corpus(tmp_path, pretokenizer, text, name)
    cfg = BPETrainerConfig(verbose=False, num_workers=2, **config)
    return BPETrainer(pretokenizer, corpus, cfg).train(), corpus


def first_merge_text(tokenizer, pretokenizer) -> str:
    return pretokenizer.tokens_repr(tokenizer.tokens[tokenizer.merge_rules[0].token_to].atomic_tokens)


def test_unibpe_prefers_rare_parents_over_frequent_pair(tmp_path, pretokenizer):
    count_tok, _ = train(tmp_path, pretokenizer, CRAFTED_TEXT, "crafted_count", additional_vocab_size=1)
    unibpe_tok, _ = train(
        tmp_path, pretokenizer, CRAFTED_TEXT, "crafted_unibpe", additional_vocab_size=1, merge_score="unibpe"
    )
    assert first_merge_text(count_tok, pretokenizer) == "pq"
    assert first_merge_text(unibpe_tok, pretokenizer) == "xy"


def test_unibpe_picks_loss_minimising_merge_at_every_step(tmp_path, pretokenizer, taylorswift_text):
    n_merges = 12
    tokenizer, corpus = train(
        tmp_path, pretokenizer, taylorswift_text[:6000], "greedy", additional_vocab_size=n_merges, merge_score="unibpe"
    )
    assert len(tokenizer.merge_rules) == n_merges

    # Replay the merges on the raw corpus and brute-force the best pair under the Unigram loss at each step.
    seqs = [(list(seq), count) for seq, count in corpus]
    for rule in tokenizer.merge_rules:
        token_counts: Counter[int] = Counter()
        pair_counts: Counter[tuple[int, int]] = Counter()
        for seq, count in seqs:
            for t in seq:
                token_counts[t] += count
            for pair in zip(seq, seq[1:]):
                if pretokenizer.bpe_merge_allowed(
                    tokenizer.tokens[pair[0]].atomic_tokens, tokenizer.tokens[pair[1]].atomic_tokens
                ):
                    pair_counts[pair] += count
        n = sum(token_counts.values())

        def merge_count(seq, pair):
            c, i = 0, 0
            while i < len(seq) - 1:
                if (seq[i], seq[i + 1]) == pair:
                    c, i = c + 1, i + 2
                else:
                    i += 1
            return c

        def delta_loss(pair):
            a, b = pair
            c = sum(merge_count(seq, pair) * count for seq, count in seqs)
            if a == b:
                return h(token_counts[a]) - h(token_counts[a] - 2 * c) - h(c) + h(n - c) - h(n)
            return (
                h(token_counts[a])
                - h(token_counts[a] - c)
                + h(token_counts[b])
                - h(token_counts[b] - c)
                - h(c)
                + h(n - c)
                - h(n)
            )

        best = min(delta_loss(p) for p in pair_counts)
        assert delta_loss(rule.tokens_from) <= best + 1e-6

        merged = []
        for seq, count in seqs:
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == rule.tokens_from:
                    out.append(rule.token_to)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            merged.append((out, count))
        seqs = merged


def test_unibpe_roundtrip_and_final_counts(tmp_path, pretokenizer, taylorswift_text):
    x_tokens = 60
    tokenizer, _ = train(
        tmp_path, pretokenizer, taylorswift_text, "roundtrip", additional_vocab_size=x_tokens, merge_score="unibpe"
    )
    assert len(tokenizer.merge_rules) == x_tokens
    assert len(tokenizer.tokens) == len(pretokenizer.atomic_tokens) + x_tokens

    ids = tokenizer.encode(taylorswift_text)
    decoded = tokenizer.decode(ids)
    original = sum(pretokenizer.pretokenize(taylorswift_text), token_array([]))
    assert sum(pretokenizer.pretokenize(decoded), token_array([])).tolist() == original.tolist()

    for token_id, count in Counter(ids).items():
        assert tokenizer.tokens[token_id].current_count == count


def test_unibpe_hybrid_switches_to_count_ranking(tmp_path, pretokenizer, taylorswift_text):
    kwargs = dict(additional_vocab_size=20, merge_score="unibpe")
    pure, _ = train(tmp_path, pretokenizer, taylorswift_text, "pure", **kwargs)
    hybrid, _ = train(tmp_path, pretokenizer, taylorswift_text, "hybrid", unibpe_merges=8, **kwargs)
    count_only, _ = train(tmp_path, pretokenizer, taylorswift_text, "count", additional_vocab_size=20)

    assert len(hybrid.merge_rules) == 20
    assert hybrid.merge_rules[:8] == pure.merge_rules[:8]
    # After the switch the hybrid ranks by pair count, so its counts at the switch are non-increasing.
    tail_counts = [hybrid.tokens[r.token_to].original_count for r in hybrid.merge_rules[8:]]
    assert tail_counts == sorted(tail_counts, reverse=True)
    assert hybrid.merge_rules != count_only.merge_rules


def test_unibpe_merges_requires_unibpe_score():
    with pytest.raises(ValueError, match="unibpe_merges"):
        BPETrainerConfig(additional_vocab_size=5, unibpe_merges=3)

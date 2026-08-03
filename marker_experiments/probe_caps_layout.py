#!/usr/bin/env python3
"""Does the caps code actually let a title-case word reuse the lowercase entry?

The shipped layout puts the code inside the delimited span:

    the  ->  <|>the<|>          The  ->  <|><^>the<|>

Those are different chunks, so the trainer learns a separate token for the title-case
form and the case duplication the codes exist to remove survives. The compression result
(-0.04%, i.e. nothing) is what a scheme that does nothing looks like.

The extcaps layout puts the code in its own chunk:

    the  ->  <|>the<|>          The  ->  <^>  <|>the<|>

so no merge can join the code to the word and the span is byte-identical to the lowercase
one. `The` is `the` plus one code token, which is what sharing an entry means.

Trains both on one corpus and counts words held in more than one cased form.

    uv run python marker_experiments/probe_caps_layout.py
"""

import os
import tempfile
from collections import defaultdict

import cyclopts

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from marker_experiments.boundary_pretokenizer import get_boundary_pretokenizer

app = cyclopts.App()


def train(pt, text, vocab):
    with tempfile.TemporaryDirectory() as d:
        corpus = PretokenizedCorpus.from_text_batches(
            name="probe", base_path=d, pretokenizer=pt,
            text_batches=iter([[text]]), num_workers=1,
        )
        return BPETrainer(
            pt, corpus, BPETrainerConfig(additional_vocab_size=vocab, num_workers=1, verbose=False)
        ).train()


def cased_forms(tokenizer, pt):
    """word -> which cased forms of it the vocabulary holds as separate entries."""
    mk, sh, cp = pt.marker_token_id, pt.shift_token_id, pt.caps_token_id
    special = {mk, sh, cp} - {None}
    forms = defaultdict(set)
    for tok in tokenizer.tokens.values():
        ids = list(tok.atomic_tokens)
        core = [i for i in ids if i not in special]
        if not core:
            continue
        txt = pt.try_decode_strict(core)
        if not txt or not txt.isalpha():
            continue
        forms[txt].add("caps" if cp in ids else ("shift" if sh in ids else "plain"))
    return forms


@app.default
def main(
    text_file: str = "tests/data/taylorswift.txt",
    vocab: int = 4000,
) -> None:
    """Train both layouts on one text and compare their case duplication.

    Args:
        text_file: Corpus to train on.
        vocab: Additional vocabulary per tokenizer.
    """
    text = open(text_file, encoding="utf-8").read()
    print(f"{text_file}: {len(text):,} chars, {vocab:,} additional vocabulary\n")

    for name in ("bnd_wpd_caps", "bnd_wpd_extcaps"):
        pt = get_boundary_pretokenizer(name)
        tok = train(pt, text, vocab)
        forms = cased_forms(tok, pt)
        dup = {w: k for w, k in forms.items() if len(k) > 1}
        n_tokens = sum(len(tok.encode(line)) for line in text.split("\n"))
        assert tok.decode(tok.encode(text)) == text, f"{name} does not round-trip"

        label = "code inside the span" if name.endswith("_caps") else "code in its own chunk"
        print(f"{name}  ({label})")
        print(f"   alphabetic word-strings   {len(forms):,}")
        print(f"   held in >1 cased form     {len(dup):,}  "
              f"({100 * len(dup) / max(len(forms), 1):.1f}%)")
        print(f"   tokens for the corpus     {n_tokens:,}")
        if dup:
            sample = sorted(dup)[-4:]
            print(f"   e.g. {', '.join(f'{w!r}:{sorted(dup[w])}' for w in sample)}")
        print(f"   'the'/'The' tokenized as  {_show(tok, pt, 'the')} / {_show(tok, pt, 'The')}")
        print()


def _show(tokenizer, pt, text):
    mk, sh, cp = pt.marker_token_id, pt.shift_token_id, pt.caps_token_id
    lab = {mk: "<|>", sh: "<^>", cp: "<^^>"}
    out = []
    for tid in tokenizer.encode(text):
        s, run = "", []
        for i in tokenizer.tokens[tid].atomic_tokens:
            if i in lab:
                if run:
                    s += pt.try_decode_strict(run) or "?"
                    run = []
                s += lab[i]
            else:
                run.append(i)
        if run:
            s += pt.try_decode_strict(run) or "?"
        out.append(s)
    return out


if __name__ == "__main__":
    app()

"""Where does bnd_w's deficit come from, and does MinGram absorb the markers?

Buckets every emitted token on the held-out English slice, for plain and bnd_w under both
trainers, so the extra tokens bnd_w spends can be attributed rather than guessed at.
"""
import json
import os
import sys
from collections import Counter

REPO = "/home/user/script_tok"
sys.path.insert(0, REPO)

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer
from script_bpe.tokenizers.mingram.model import MinGramModel
from marker_experiments.boundary_pretokenizer import get_boundary_pretokenizer

TOK = os.path.join(REPO, "marker_experiments", "tokenizers")
texts = json.load(open(os.path.join(REPO, "marker_experiments", "eval_texts", "en.json")))
chars = sum(map(len, texts))

MK = "<|>"


def load(arm, trainer):
    cls = MinGramModel if trainer == "mingram" else BPETokenizer
    t = cls.load(os.path.join(TOK, f"mg250_en_{arm}_{trainer}_32k.json.gz"))
    pt = get_pretokenizer("scriptenc3_cb") if arm == "plain" else get_boundary_pretokenizer(arm)
    return t, pt


def token_text(t, pt, tid):
    """Decoded text of one vocabulary entry, with markers shown in place."""
    ids = list(t.tokens[tid].atomic_tokens)
    m = getattr(pt, "marker_token_id", None)
    core = [i for i in ids if i != m]
    body = pt.try_decode_strict(core) if core else ""
    pre = MK if m is not None and ids and ids[0] == m else ""
    post = MK if m is not None and len(ids) > 1 and ids[-1] == m else ""
    return pre + (body or "") + post


def bucket(s):
    core = s.replace(MK, "")
    if not core:
        return "marker only"
    if core.strip(" ") == "":
        return "whitespace only"
    if any(c.isalpha() for c in core):
        return "has letters"
    if any(c.isdigit() for c in core):
        return "has digits"
    return "punctuation etc"


print(f"held-out English: {chars:,} chars, {len(texts)} docs\n")
summary = {}
for trainer in ("bpe", "mingram"):
    for arm in ("plain", "bnd_w"):
        t, pt = load(arm, trainer)
        texts_by_id = {tid: token_text(t, pt, tid) for tid in t.tokens}
        counts = Counter()
        total = 0
        marker_bearing = 0
        for doc in texts:
            for tid in t.encode(doc):
                s = texts_by_id[tid]
                counts[bucket(s)] += 1
                total += 1
                if MK in s and s != MK:
                    marker_bearing += 1
        summary[(trainer, arm)] = (total, counts, marker_bearing)
        print(f"{trainer:<8} {arm:<7} {total:>9,} tokens  {chars/total:.4f} ch/tok")
        for k, v in counts.most_common():
            print(f"           {k:<16} {v:>9,}  {100*v/total:5.2f}%")
        if marker_bearing:
            print(f"           {'marker absorbed':<16} {marker_bearing:>9,}  "
                  f"{100*marker_bearing/total:5.2f}% of tokens carry a marker inside a piece")
        print()

print("=== bnd_w minus plain, by bucket (extra tokens spent) ===")
for trainer in ("bpe", "mingram"):
    tp, cp, _ = summary[(trainer, "plain")]
    tb, cb, mb = summary[(trainer, "bnd_w")]
    print(f"\n{trainer}:  {tp:,} -> {tb:,}   ({tb-tp:+,} tokens, {100*(tb-tp)/tp:+.2f}%)")
    for k in sorted(set(cp) | set(cb), key=lambda k: -(cb[k] - cp[k])):
        d = cb[k] - cp[k]
        share = 100 * d / (tb - tp) if tb != tp else 0
        print(f"   {k:<16} {cp[k]:>9,} -> {cb[k]:>9,}   {d:>+9,}  ({share:5.1f}% of the deficit)")

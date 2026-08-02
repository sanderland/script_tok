"""Does the merge constraint do anything, given how the pretokenizer chunks?

`bpe_merge_allowed` refuses a merge whose left part ends in a marker and whose right part
begins with one, to stop BPE learning '<|>the<|><|>'. But the pretokenizer already emits
each unit as its own chunk, so the two touching markers straddle a chunk boundary, and BPE
merges only within a chunk. Train with and without the rule and compare.
"""
import tempfile

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from marker_experiments.boundary_pretokenizer import (
    BoundaryScriptPretokenizer,
    BoundaryScriptPretokenizerConfig,
)
from script_bpe.pretokenize.pretokenizer import ScriptPretokenizer
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3


class UnconstrainedConfig(BoundaryScriptPretokenizerConfig):
    cls: str = "UnconstrainedBoundaryPretokenizer"


class UnconstrainedBoundaryPretokenizer(BoundaryScriptPretokenizer, config_type=UnconstrainedConfig):
    """Identical, minus the touching-marker merge rule."""

    def bpe_merge_allowed(self, a, b) -> bool:
        return ScriptPretokenizer.bpe_merge_allowed(self, a, b)


def vocab(pt, text, n):
    with tempfile.TemporaryDirectory() as d:
        corpus = PretokenizedCorpus.from_text_batches(
            name="probe", base_path=d, pretokenizer=pt,
            text_batches=iter([[text]]), num_workers=1,
        )
        t = BPETrainer(pt, corpus, BPETrainerConfig(additional_vocab_size=n, num_workers=1,
                                                   verbose=False)).train()
    mk = pt.marker_token_id
    out, bad = set(), []
    for tok in t.tokens.values():
        ids = tuple(tok.atomic_tokens)
        out.add(ids)
        for i in range(len(ids) - 1):
            if ids[i] == mk and ids[i + 1] == mk:
                bad.append(ids)
                break
    return out, bad


if __name__ == "__main__":
    text = open("tests/data/taylorswift.txt").read()
    kw = dict(script_config=ScriptEncodingV3, boundary_targets=("word", "punct", "digit"))
    on = BoundaryScriptPretokenizer(BoundaryScriptPretokenizerConfig(**kw))
    off = UnconstrainedBoundaryPretokenizer(UnconstrainedConfig(**kw))

    v_on, bad_on = vocab(on, text, 2000)
    v_off, bad_off = vocab(off, text, 2000)

    print(f"vocabulary with the rule   : {len(v_on):,}  tokens containing '<|><|>': {len(bad_on)}")
    print(f"vocabulary without the rule: {len(v_off):,}  tokens containing '<|><|>': {len(bad_off)}")
    print(f"identical vocabularies     : {v_on == v_off}")
    if v_on != v_off:
        print(f"  only with the rule   : {len(v_on - v_off)}")
        print(f"  only without the rule: {len(v_off - v_on)}")

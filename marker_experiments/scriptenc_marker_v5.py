"""Boundary-marker pretokenizer, v5 = v4 + digits are markable.

Measured motivation (held-out prose and code evals, under v4):

    domain  single spaces   elided   not elided   of which digit-adjacent
    code       42,242        87.5%      12.5%           97.9%
    prose     287,220        96.7%       3.3%           98.7%

Digits account for ~98% of every single space v4 fails to elide, and code has
3.8x prose's rate of them -- which is why v4's code penalty ran ~2x its prose
penalty. It was never indentation: multi-space runs are ~1.2% of emitted tokens
and identical under the baseline and v4.

Why digits were excluded to begin with: script_category_v3 folds L/M -> LM,
Z/Cc -> ZC, So -> So and P/S/Cf -> PSF, but leaves category N alone. Digits
therefore land in (ALL, "N") blocks, which are absent from script_cat_with_spaces
and so were classified OTHER (non-markable) by v4.

v5 marks digits with the SAME asymmetric rule as punctuation -- a marker only on
a side whose single space was actually elided, never unconditionally:

    'x = 1'  ->  <|>x<|>  <|>=<|>  <|>1     (both spaces elided)
    'a1'     ->  <|>a<|>  1                 (no space, no marker)
    '1a'     ->  1  <|>a<|>                 (no space, no marker)

The invariant survives -- two touching <|> still means exactly one elided space --
because only word spans are ever marked unconditionally, and word|word adjacency
is already handled in v4.

Digits are a closed set (10 per script), so variant cost stays bounded, as it
does for punctuation. Deliberately NOT extended to Han/emoji/other scripts:
those are open sets where per-unit marker variants could multiply, and they
account for only ~2% of non-elided spaces.
"""

from script_bpe.pretokenize.pretokenizer import ScriptPretokenizerConfig

from scriptenc_marker_v4 import MarkerV4Pretokenizer


class MarkerV5PretokenizerConfig(ScriptPretokenizerConfig):
    cls: str = "MarkerV5Pretokenizer"


class MarkerV5Pretokenizer(MarkerV4Pretokenizer, config_type=MarkerV5PretokenizerConfig):
    def _extra_markable_script_ids(self) -> set:
        # category "N" (Nd/Nl/No) survives script_category_v3's supercategory folding
        # untouched, with script rewritten to ALL as for all non-letters.
        return {b.script_id for b in self.config.script_config.blocks if b.category == "N"}

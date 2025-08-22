import copy
import gzip
import heapq
import json
import os
from dataclasses import dataclass

import tabulate

from script_bpe.pretokenize import Pretokenizer, export_pretokenizer, load_pretokenizer
from script_bpe.utils import InputTokenSeq, TokenSeq, token_array
from script_bpe.tokenizers.base import BaseToken, BaseTokenizer


@dataclass(slots=True)
class MergeRule:
    tokens_from: tuple[int, int]
    token_to: int

    def save_dict(self):
        return dict(tokens_from=self.tokens_from, token_to=self.token_to)


@dataclass(slots=True)
class Token(BaseToken):
    current_count: int = 0
    original_count: int = 0

    def pretty_repr(self, pretokenizer):
        """Return a pretty string for the token, with its base tokens and readable representation"""
        # {self.atomic_tokens.tolist()}
        return f"Token({self.id}, {repr(pretokenizer.tokens_repr(self.atomic_tokens))})"

    def report_dict(self, pretokenizer):
        """Return fields for printing to a table for human inspection"""
        return dict(
            id=self.id,
            vocab=pretokenizer.tokens_repr(self.atomic_tokens),
            original_count=self.original_count,
            final_count=self.current_count,
        )

    def to_dict(self) -> dict:
        return dict(
            id=self.id,
            atomic_tokens=list(self.atomic_tokens),
            original_count=self.original_count,
            current_count=self.current_count,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Token":
        return cls(
            id=d["id"],
            atomic_tokens=token_array(d["atomic_tokens"]),
            original_count=d.get("original_count", 0),
            current_count=d.get("current_count", 0),
        )


class BPETokenizer(BaseTokenizer):
    VERSION = "sebpe-v1"
    REPORT_TITLE = "BPE Tokenizer Report"

    def __init__(self, merge_rules, pretokenizer: Pretokenizer, metadata=None, tokens: dict[int, Token] | None = None):
        self.pretokenizer = pretokenizer
        self.merge_rules = merge_rules
        self.metadata = metadata or {}
        self._merge_rules_dict: dict[tuple[int, int], tuple[float, int]] = {}
        if tokens is not None:
            self.tokens = tokens
            # build merge rule index
            for mi, mr in enumerate(self.merge_rules):
                self._merge_rules_dict[(mr.tokens_from[0], mr.tokens_from[1])] = (mi, mr.token_to)
        else:
            self._build_vocab()

    @classmethod
    def load(cls, file):
        open_func = gzip.open if file.endswith(".gz") else open
        with open_func(file, "rt") as f:
            data = json.load(f)

        merge_rules = [MergeRule(**mr) for mr in data["merge_rules"]]
        pretokenizer = load_pretokenizer(data["pretokenizer"])
        tokens = None
        if "tokens" in data:
            toks = [Token.from_dict(t) for t in data["tokens"]]
            tokens = {t.id: t for t in toks}
        return cls(merge_rules=merge_rules, pretokenizer=pretokenizer, metadata=data.get("metadata", {}), tokens=tokens)

    def _build_vocab(self):
        # single bytes
        self.tokens = {i: Token(i, token_array([i])) for i in self.pretokenizer.atomic_tokens.copy()}
        # build other tokens from merge rules
        for mi, mr in enumerate(self.merge_rules):
            to_id = mr.token_to
            from_a, from_b = mr.tokens_from
            self.tokens[to_id] = Token(
                id=to_id, atomic_tokens=self.tokens[from_a].atomic_tokens + self.tokens[from_b].atomic_tokens
            )
            self._merge_rules_dict[(from_a, from_b)] = (mi, to_id)

    def decode(self, ids: InputTokenSeq, errors="replace") -> str:
        atomic_tokens = [bt for i in ids for bt in self.tokens[i].atomic_tokens]
        return self.pretokenizer.decode(atomic_tokens, errors=errors)

    def _encode_chunk(self, ids_arr: TokenSeq) -> list[int]:
        len_ids = len(ids_arr)
        merge_heap = [
            (mr[0], i, ids_arr[i], i + 1, ids_arr[i + 1], mr[1])
            for i in range(len(ids_arr) - 1)
            if (mr := self._merge_rules_dict.get((ids_arr[i], ids_arr[i + 1])))
        ]
        heapq.heapify(merge_heap)
        while merge_heap:
            # pop the lowest merge index
            _, from_a, val_a, from_b, val_b, to_id = heapq.heappop(merge_heap)
            # if the merge rule is not valid anymore, skip it
            if ids_arr[from_a] != val_a or ids_arr[from_b] != val_b:
                continue
            ids_arr[from_a] = to_id
            ids_arr[from_b] = -1
            from_next = from_b + 1
            while from_next < len_ids and ids_arr[from_next] == -1:
                from_next += 1
            if from_next < len_ids:
                # check if the next merge rule is valid
                if mr := self._merge_rules_dict.get((to_id, ids_arr[from_next])):
                    heapq.heappush(merge_heap, (mr[0], from_a, to_id, from_next, ids_arr[from_next], mr[1]))
            from_prev = from_a - 1
            while from_prev >= 0 and ids_arr[from_prev] == -1:
                from_prev -= 1
            if from_prev >= 0:
                # check if the previous merge rule is valid
                if mr := self._merge_rules_dict.get((ids_arr[from_prev], to_id)):
                    heapq.heappush(merge_heap, (mr[0], from_prev, ids_arr[from_prev], from_a, to_id, mr[1]))

        return [i for i in ids_arr if i >= 0]

    def encode(self, text: str) -> TokenSeq:
        chunks = self.pretokenizer.pretokenize(text)
        return token_array([t for chunk in chunks for t in self._encode_chunk(chunk)])

    def save(self, file_path) -> str:
        """Save the tokenizer to a file using the given file prefix."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        open_func = gzip.open if file_path.endswith(".gz") else open
        with open_func(file_path, "wt") as f:
            json.dump(
                dict(
                    info=dict(version=self.VERSION),
                    merge_rules=[mr.save_dict() for mr in self.merge_rules],
                    tokens=[t.to_dict() for t in self.tokens.values()],
                    metadata={k: v for k, v in self.metadata.items() if k != "tokens"},
                    pretokenizer=export_pretokenizer(self.pretokenizer),
                ),
                f,
                indent=2,
            )
        return file_path

    def stats(self, n_longest=50) -> dict:
        """Compute and return statistics about the tokenizer."""
        s = super().stats(n_longest=n_longest)
        s["num_merge_rules"] = len(self.merge_rules)
        # Last merge original count from tokens
        s["last_merge_count"] = (
            self.tokens[self.merge_rules[-1].token_to].original_count if self.merge_rules else 0
        )
        # Optional: derive entropy-like metrics from current counts
        counts = [t.current_count for t in self.tokens.values() if t.current_count > 0]
        n = sum(counts)
        if n > 0:
            import math as _math
            probs = [c / n for c in counts]
            shannon = -sum(p * _math.log2(p) for p in probs)
            def _renyi(ps: list[float], alpha: float) -> float:
                if alpha == 1.0:
                    return shannon
                if alpha == float("inf"):
                    maxp = max(ps) if ps else 0.0
                    return -_math.log2(maxp) if maxp > 0.0 else 0.0
                if alpha == 0.0:
                    k = len(ps)
                    return _math.log2(k) if k > 0 else 0.0
                ssum = sum(p ** alpha for p in ps)
                return _math.log2(ssum) / (1.0 - alpha) if ssum > 0.0 else 0.0
            alphas = (0.5, 1.0, 2.0, float("inf"))
            s["train_renyi_bits"] = {("inf" if a == float("inf") else str(a)): _renyi(probs, a) for a in alphas}
            s["train_shannon_bits"] = shannon
        return s

    def report_details(self, n_longest: int = 20) -> dict[str, list[dict]]:
        merge_rows = []
        for mr in (mr.save_dict() for mr in self.merge_rules):
            merge_rows.append(
                {
                    "tokens_from": mr["tokens_from"],
                    "token_to": mr["token_to"],
                    "vocab_from": [
                        self.pretokenizer.tokens_repr(self.tokens[t].atomic_tokens) for t in mr["tokens_from"]
                    ],
                    "vocab_to": repr(self.pretokenizer.tokens_repr(self.tokens[mr["token_to"]].atomic_tokens)),
                    "count": self.tokens[mr["token_to"]].original_count,
                }
            )
        token_rows = [
            {
                "id": t.id,
                "vocab": repr(self.pretokenizer.tokens_repr(t.atomic_tokens)),
                "original_count": t.original_count,
                "final_count": t.current_count,
            }
            for t in self.tokens.values()
            if t.id not in self.pretokenizer.atomic_tokens
        ]
        return {
            f"Details for {len(self.merge_rules):,d} merge rules": merge_rows,
            f"Details for {len(token_rows):,d} tokens": token_rows,
        }

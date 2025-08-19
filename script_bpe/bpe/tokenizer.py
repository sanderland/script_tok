import array
import copy
import gzip
import heapq
import json
import os
from dataclasses import dataclass

import tabulate

from script_bpe.pretokenize import Pretokenizer, export_pretokenizer, make_pretokenizer
from script_bpe.utils import InputTokenSeq, TokenSeq, token_array


@dataclass(slots=True)
class MergeRule:
    tokens_from: tuple[int, int]
    token_to: int

    def save_dict(self):
        return dict(tokens_from=self.tokens_from, token_to=self.token_to)


@dataclass(slots=True)
class Token:
    id: int
    base_tokens: TokenSeq
    current_count: int = 0
    original_count: int = 0

    def pretty_repr(self, pretokenizer):
        """Return a pretty string for the token, with its base tokens and readable representation"""
        # {self.base_tokens.tolist()}
        return f"Token({self.id}, {repr(pretokenizer.tokens_to_readable_string(self.base_tokens))})"

    def report_dict(self, pretokenizer):
        """Return fields for printing to a table for human inspection"""
        return dict(
            id=self.id,
            vocab=pretokenizer.tokens_to_readable_string(self.base_tokens),
            original_count=self.original_count,
            final_count=self.current_count,
        )


class BPETokenizer:
    VERSION = "sebpe-v1"

    def __init__(self, merge_rules, pretokenizer: Pretokenizer, metadata=None):
        self.pretokenizer = pretokenizer
        self.merge_rules = merge_rules
        self.metadata = metadata or {}
        self._merge_rules_dict = {}
        self._build_vocab()

    @classmethod
    def load(cls, file):
        open_func = gzip.open if file.endswith(".gz") else open
        with open_func(file, "rt") as f:
            data = json.load(f)

        merge_rules = [MergeRule(**mr) for mr in data["merge_rules"]]
        pretokenizer = make_pretokenizer(data["pretokenizer"])
        return cls(merge_rules=merge_rules, pretokenizer=pretokenizer, metadata=data.get("metadata", {}))

    def _build_vocab(self):
        # single bytes
        self.tokens = {i: Token(i, token_array([i])) for i in self.pretokenizer.base_tokens.copy()}
        # build other tokens from merge rules
        for mi, mr in enumerate(self.merge_rules):
            to_id = mr.token_to
            from_a, from_b = mr.tokens_from
            self.tokens[to_id] = Token(
                id=to_id, base_tokens=self.tokens[from_a].base_tokens + self.tokens[from_b].base_tokens
            )
            self._merge_rules_dict[(from_a, from_b)] = (mi, to_id)

    def decode(self, ids: InputTokenSeq, errors="replace") -> str:
        base_tokens = [bt for i in ids for bt in self.tokens[i].base_tokens]
        return self.pretokenizer.decode(base_tokens, errors=errors)

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
                    metadata=self.metadata,
                    pretokenizer=export_pretokenizer(self.pretokenizer),
                ),
                f,
                indent=2,
            )
        return file_path

    def stats(self, n_longest=50) -> dict:
        """Compute and return statistics about the tokenizer."""
        num_merge_rules = len(self.merge_rules)
        num_tokens = len(self.tokens)
        metadata_tokens = {t["id"]: t for t in copy.deepcopy(self.metadata.get("tokens", []))}
        last_merge_count = metadata_tokens[self.merge_rules[-1].token_to]["original_count"]
        longest_tokens_base_tokens = sorted(self.tokens.values(), key=lambda t: -len(t.base_tokens))[:n_longest]
        longest_tokens_chars = sorted(self.tokens.values(), key=lambda t: -len(self.decode([t.id])))[:n_longest]

        avg_token_length_bt = sum(len(t.base_tokens) for t in self.tokens.values()) / num_tokens
        is_undecodable = {
            id: "�" in self.pretokenizer.decode(t.base_tokens, errors="replace")
            and (
                self.pretokenizer.decode(t.base_tokens, errors="replace").count("�")
                - self.pretokenizer.tokens_to_readable_string(t.base_tokens).count("�")
            )
            > 0
            for id, t in self.tokens.items()
            if id not in self.pretokenizer.base_tokens
        }
        num_undecodeable = sum(is_undecodable.values())
        return dict(
            num_merge_rules=num_merge_rules,
            num_tokens=num_tokens,
            num_undecodeable=num_undecodeable,
            last_merge_count=last_merge_count,
            longest_tokens_base_tokens=longest_tokens_base_tokens,
            longest_tokens_chars=longest_tokens_chars,
            avg_token_length_bt=avg_token_length_bt,
        )

    def report(self):
        """Return markdown formatted report with info."""
        stats = self.stats()
        metadata_tokens = {t["id"]: t for t in copy.deepcopy(self.metadata.get("tokens", []))}

        # Generate report
        report = f"# Tokenizer report\n"
        report += f"\n## Statistics\n\n"
        report += f"- Number of merge rules trained: {stats['num_merge_rules']}\n"
        report += f"- Total number of tokens: {stats['num_tokens']}\n"
        report += f"- Last merge frequency: {stats['last_merge_count']}\n"
        report += f"- Average token length: {stats['avg_token_length_bt']:.4f} base tokens\n"
        report += f"- Number of undecodeable tokens: {stats['num_undecodeable']}\n"

        for longest_type, longest_tokens in [
            ("base tokens", stats["longest_tokens_base_tokens"]),
            ("characters", stats["longest_tokens_chars"]),
        ]:
            report += f"\n## Longest tokens by {longest_type}\n\n"
            for t in longest_tokens:
                report += f"- Token {t.id:6d} consists of {len(t.base_tokens):3d} {longest_type}: {self.pretokenizer.tokens_to_readable_string(t.base_tokens)!r}\n"

        # Merge rules
        report += f"\n## Details for {len(self.merge_rules):,d} merge rules\n\n"
        merge_rules = [mr.save_dict() for mr in self.merge_rules]
        for mr in merge_rules:
            mr["vocab_from"] = [
                self.pretokenizer.tokens_to_readable_string(self.tokens[t].base_tokens) for t in mr["tokens_from"]
            ]
            mr["vocab_to"] = repr(self.pretokenizer.tokens_to_readable_string(self.tokens[mr["token_to"]].base_tokens))
            mr["count"] = metadata_tokens[mr["token_to"]]["original_count"]
        report += tabulate.tabulate(merge_rules, headers="keys", tablefmt="github")

        # tokens
        for t in metadata_tokens.values():
            t["vocab"] = repr(t["vocab"])
        non_base_tokens = [t for t in metadata_tokens.values() if t["id"] not in self.pretokenizer.base_tokens]
        report += f"\n\n## Details for {len(metadata_tokens):,d} tokens\n\n"
        report += tabulate.tabulate(non_base_tokens, headers="keys", tablefmt="github")

        # Metadata
        report += f"\n\n## Metadata\n\n"
        report += tabulate.tabulate(
            [[k, v] for k, v in self.metadata.items() if k != "tokens"], headers=["key", "value"], tablefmt="github"
        )
        return report

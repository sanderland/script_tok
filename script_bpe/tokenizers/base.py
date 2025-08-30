from dataclasses import dataclass
from typing import Iterable, Mapping
from collections import Counter
import math


from pydantic import BaseModel

from script_bpe.utils import InputTokenSeq, TokenSeq, create_logger
from script_bpe.pretokenize import Pretokenizer
from script_bpe.corpus import PretokenizedCorpus


@dataclass(slots=True)
class BaseToken:
    id: int
    atomic_tokens: TokenSeq


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


class BaseTokenizer:
    VERSION: str
    REPORT_TITLE: str = "Tokenizer Report"
    pretokenizer: Pretokenizer
    tokens: Mapping[int, BaseToken]

    def encode(self, text: str, return_tokens: bool = False) -> TokenSeq | list[BaseToken]:
        raise NotImplementedError

    def decode(self, ids: InputTokenSeq, errors: str = "replace") -> str:
        raise NotImplementedError

    def save(self, file_path: str) -> str:
        raise NotImplementedError

    @classmethod
    def load(cls, file: str):
        raise NotImplementedError

    def report_details(self, n_longest: int = 20) -> dict[str, list[dict]]:
        """Return tokenizer-specific, detailed sections for the report.

        Each entry maps a section title to a list of table rows (dicts). The
        base report() will render these using tabulate with headers='keys'.
        """
        return {}

    def report(self, n_longest: int = 20) -> str:
        from tabulate import tabulate as _tab

        stats = self.stats(n_longest=n_longest)
        report_lines: list[str] = [f"# {self.REPORT_TITLE}", "", "## Statistics", ""]

        # Common statistics
        report_lines.extend(
            [
                f"- Total number of tokens: {stats['num_tokens']}",
                f"- Average token length (atomic tokens): {stats['avg_token_length_bt']:.4f}",
                f"- Average characters per token: {stats['avg_char_length']:.4f}",
                f"- Atomic tokens (single token): {stats['num_atomic_tokens']}",
                f"- Multi-token sequences: {stats['num_multi_tokens']}",
                f"- Undecodable tokens (strict; excluding atomic): {stats['num_undecodable']}",
                f"- Inherited-start multi-char tokens: {stats.get('num_inherited_start_violations', 0)}",
            ]
        )
        # Optional statistics (if present)
        if 'num_merge_rules' in stats:
            report_lines.append(f"- Number of merge rules trained: {stats['num_merge_rules']}")
        if 'last_merge_count' in stats:
            report_lines.append(f"- Last merge frequency: {stats['last_merge_count']}")

        # Generic details for all tokenizers
        longest_atomic_rows = [
            {"ID": t.id, "Atomic Tokens": len(t.atomic_tokens), "Text": repr(self.pretokenizer.tokens_repr(t.atomic_tokens))}
            for t in stats["longest_tokens_by_atomic"]
        ]
        longest_chars_rows = [
            {"ID": t.id, "Characters": len(self.decode([t.id])), "Text": repr(self.pretokenizer.tokens_repr(t.atomic_tokens))}
            for t in stats["longest_tokens_by_chars"]
        ]
        report_lines.extend([
            "",
            "## Longest tokens by atomic tokens",
            "",
            _tab(longest_atomic_rows, headers="keys", tablefmt="github"),
            "",
            "## Longest tokens by characters",
            "",
            _tab(longest_chars_rows, headers="keys", tablefmt="github"),
        ])

        # Detailed sections provided by subclass
        details = self.report_details(n_longest=n_longest)
        for title, rows in details.items():
            report_lines.append("")
            report_lines.append(f"## {title}")
            report_lines.append("")
            if rows:
                report_lines.append(_tab(rows, headers="keys", tablefmt="github"))
                report_lines.append("")

        return "\n".join(report_lines)

    def stats(self, n_longest: int = 20) -> dict:
        num_tokens = len(self.tokens)

        num_atomic_tokens = sum(1 for t in self.tokens.values() if len(t.atomic_tokens) == 1)
        num_multi_tokens = sum(1 for t in self.tokens.values() if len(t.atomic_tokens) > 1)
        avg_token_length_bt = sum(len(t.atomic_tokens) for t in self.tokens.values()) / num_tokens
        char_lengths = [len(self.decode([t.id])) for t in self.tokens.values()]
        avg_char_length = sum(char_lengths) / num_tokens if num_tokens else 0.0

        # Strict undecodable and inherited-start violations in a single pass
        num_undecodable = 0
        num_inherited_start_violations = 0
        for t in self.tokens.values():
            if len(t.atomic_tokens) <= 1:
                continue
            text = self.pretokenizer.try_decode_strict(t.atomic_tokens)
            if text is None:
                num_undecodable += 1
                continue
            if self.pretokenizer.violates_inherited_order(text):
                num_inherited_start_violations += 1

        longest_by_atomic = sorted(self.tokens.values(), key=lambda t: -len(t.atomic_tokens))[:n_longest]
        longest_by_chars = sorted(self.tokens.values(), key=lambda t: -len(self.decode([t.id])))[:n_longest]

        return dict(
            num_tokens=num_tokens,
            num_atomic_tokens=num_atomic_tokens,
            num_multi_tokens=num_multi_tokens,
            num_undecodable=num_undecodable,
            num_inherited_start_violations=num_inherited_start_violations,
            avg_token_length_bt=avg_token_length_bt,
            avg_char_length=avg_char_length,
            longest_tokens_by_atomic=longest_by_atomic,
            longest_tokens_by_chars=longest_by_chars,
        )

    def corpus_performance(
        self, corpus: PretokenizedCorpus, alphas: Iterable[float] = (0.5, 1.0, 2.0, float("inf"))
    ) -> dict:
        total_char_len = 0
        total_base_len = 0
        total_tokens_len = 0
        token_freqs: Counter[int] = Counter()

        for atomic_tokens, count in corpus:
            text = self.pretokenizer.decode(atomic_tokens)
            encoded = self.encode(text)

            total_char_len += len(text) * count
            total_base_len += len(atomic_tokens) * count
            total_tokens_len += len(encoded) * count

            if encoded:
                ct = Counter(int(t) for t in encoded)
                token_freqs.update({tid: count * c for tid, c in ct.items()})

        def _shannon_bits(freqs: Counter[int]) -> float:
            n = sum(freqs.values())
            if n == 0:
                return 0.0
            h = 0.0
            for c in freqs.values():
                p = c / n
                if p > 0.0:
                    h -= p * math.log2(p)
            return h

        def _renyi_bits(freqs: Counter[int], alpha: float) -> float:
            n = sum(freqs.values())
            if n == 0:
                return 0.0
            if alpha == 1.0:
                return _shannon_bits(freqs)
            if alpha == float("inf"):
                maxp = max((c / n) for c in freqs.values()) if freqs else 0.0
                return -math.log2(maxp) if maxp > 0.0 else 0.0
            if alpha == 0.0:
                k = sum(1 for c in freqs.values() if c > 0)
                return math.log2(k) if k > 0 else 0.0
            s = 0.0
            for c in freqs.values():
                p = c / n
                if p > 0.0:
                    s += p**alpha
            if s == 0.0:
                return 0.0
            return math.log2(s) / (1.0 - alpha)

        renyi = {("inf" if a == float("inf") else str(a)): _renyi_bits(token_freqs, a) for a in alphas}

        return {
            "total_char_len": total_char_len,
            "total_base_len": total_base_len,
            "total_tokens_len": total_tokens_len,
            "tokens_per_char": safe_div(total_tokens_len, total_char_len),
            "tokens_per_base": safe_div(total_tokens_len, total_base_len),
            "chars_per_token": safe_div(total_char_len, total_tokens_len),
            "base_per_token": safe_div(total_base_len, total_tokens_len),
            "renyi_bits": renyi,
            "shannon_bits": renyi.get("1.0", 0.0),
            "nonzero_vocab": sum(1 for c in token_freqs.values() if c > 0),
        }


class TrainerConfig(BaseModel):
    verbose: bool = True
    additional_vocab_size: int
    num_workers: int = 4


class BaseTrainer:
    def __init__(self, pretokenizer: Pretokenizer, corpus: PretokenizedCorpus, config: TrainerConfig):
        self.pretokenizer = pretokenizer
        self.corpus = corpus
        self.config = config
        self.logger = create_logger(self.__class__.__name__, config.verbose)

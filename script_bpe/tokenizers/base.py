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

    def report(self) -> str:
        raise NotImplementedError

    def stats(self, n_longest: int = 20) -> dict:
        num_tokens = len(self.tokens)

        def _is_undecodable(token: "BaseToken") -> bool:
            decoded = self.pretokenizer.decode(token.atomic_tokens, errors="replace")
            robust = self.pretokenizer.tokens_repr(token.atomic_tokens)
            return decoded.count("�") > robust.count("�")

        num_atomic_tokens = sum(1 for t in self.tokens.values() if len(t.atomic_tokens) == 1)
        num_multi_tokens = sum(1 for t in self.tokens.values() if len(t.atomic_tokens) > 1)
        avg_token_length_bt = sum(len(t.atomic_tokens) for t in self.tokens.values()) / num_tokens
        char_lengths = [len(self.decode([t.id])) for t in self.tokens.values()]
        avg_char_length = sum(char_lengths) / num_tokens if num_tokens else 0.0
        num_undecodable = sum(1 for t in self.tokens.values() if _is_undecodable(t))

        longest_by_atomic = sorted(self.tokens.values(), key=lambda t: -len(t.atomic_tokens))[:n_longest]
        longest_by_chars = sorted(self.tokens.values(), key=lambda t: -len(self.decode([t.id])))[:n_longest]

        return dict(
            num_tokens=num_tokens,
            num_atomic_tokens=num_atomic_tokens,
            num_multi_tokens=num_multi_tokens,
            num_undecodable=num_undecodable,
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

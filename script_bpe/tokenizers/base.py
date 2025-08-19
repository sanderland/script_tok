from dataclasses import dataclass

from script_bpe.utils import TokenSeq


@dataclass(slots=True)
class BaseToken:
	id: int
	atomic_tokens: TokenSeq



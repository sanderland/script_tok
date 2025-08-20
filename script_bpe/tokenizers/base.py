from dataclasses import dataclass
from typing import Protocol, Iterable

from pydantic import BaseModel

from script_bpe.utils import TokenSeq, create_logger
from script_bpe.pretokenize import Pretokenizer
from script_bpe.corpus import PretokenizedCorpus


@dataclass(slots=True)
class BaseToken:
	id: int
	atomic_tokens: TokenSeq


class BaseTokenizer(Protocol):
	VERSION: str
	pretokenizer: Pretokenizer

	def encode(self, text: str) -> TokenSeq: ...
	def decode(self, ids: Iterable[int], errors: str = "replace") -> str: ...
	def save(self, file_path: str) -> str: ...
	@classmethod
	def load(cls, file: str): ...
	def report(self) -> str: ...


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


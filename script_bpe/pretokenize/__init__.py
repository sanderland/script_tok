from typing import Callable

class BasePretokenizer: # TODO remove
    pass

def export_pretokenizer(pretok: BasePretokenizer) -> dict:
    ...

def make_pretokenizer(config: dict) -> BasePretokenizer:
    ...

def get_pretokenizer(name: str) -> Callable[[], BasePretokenizer]:
    ...

PRETOKENIZER_REGISTRY = {}
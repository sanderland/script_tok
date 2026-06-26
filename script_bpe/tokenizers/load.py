import gzip
import json

from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.convextok import ConvexTokModel
from script_bpe.tokenizers.mingram import MinGramModel
from script_bpe.tokenizers.pathpiece import PathPieceModel
from script_bpe.tokenizers.unigram import UnigramModel


def detect_tokenizer_model_class(path: str):
    """Detect the native tokenizer model class from a saved file."""
    open_func = gzip.open if path.endswith(".gz") else open
    with open_func(path, "rt") as f:
        data = json.load(f)

    version = data["info"]["version"]
    if version.startswith("sebpe"):
        return BPETokenizer
    if version.startswith("seunigram"):
        return UnigramModel
    if version.startswith("semingram"):
        return MinGramModel
    if version.startswith("sepathpiece"):
        return PathPieceModel
    if version.startswith("seconvextok"):
        return ConvexTokModel
    raise ValueError(f"Unsupported script_bpe tokenizer version: {version}")


def load_tokenizer(path: str, model_type: str | None = None, reindex: bool = False):
    """Load a native `script_bpe` tokenizer from disk.

    Args:
            path: Path to the saved tokenizer file.
            model_type: Optional explicit model type. When omitted, the helper
                inspects the saved file's `info.version`.
            reindex: Passed through to Unigram/MinGram `load()` so callers can
                opt into dense model-token ids. BPE ignores this flag because
                its native loader has no reindexing mode.
    """
    model_classes = {
        "bpe": BPETokenizer,
        "unigram": UnigramModel,
        "mingram": MinGramModel,
        "pathpiece": PathPieceModel,
        "convextok": ConvexTokModel,
    }
    if model_type is not None:
        if model_type not in model_classes:
            raise ValueError(f"Unknown script_bpe model type: {model_type}")
        model_class = model_classes[model_type]
    else:
        model_class = detect_tokenizer_model_class(path)

    if model_class in (UnigramModel, MinGramModel, PathPieceModel, ConvexTokModel):
        return model_class.load(path, reindex=reindex)
    return model_class.load(path)

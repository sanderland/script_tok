from pathlib import Path
import json
import math

from paper_utils.hybrid.train_hybrid import get_model_path as get_hybrid_model_path
from paper_utils.unigram.train_hyperparameters import (
    ADDITIONAL_VOCAB_SIZE,
    DEFAULTS,
    get_model_path as get_unigram_model_path,
)
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
PAPER_DIR = REPO_ROOT / "results/mingram_paper"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_EXTRA_DIR = PAPER_DIR / "extra"
UNIGRAM_RESULTS_DIR = REPO_ROOT / "results/unigram_sweeps"
UNIGRAM_SCRIPTENC_RESULTS_DIR = UNIGRAM_RESULTS_DIR / "scriptenc_cb"

DEFAULT_OVERSHOOT_FACTOR = 1.1
FSP_OVERRIDES = {"flat_score_prune": True, "pre_final_vocab_factor": 1.0}
FSP_PARAMS = {**DEFAULTS, **FSP_OVERRIDES}
MORPHALIGN_SCORE_SCALE = 100.0


def _app_name(filename: str, appendix: bool) -> str:
    if appendix and not filename.startswith("app_"):
        return f"app_{filename}"
    return filename


def paper_figure_path(filename: str, *, appendix: bool = False, extra: bool = False) -> Path:
    base_dir = PAPER_EXTRA_DIR if extra else PAPER_FIGURES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / _app_name(filename, appendix)


def paper_table_path(filename: str, *, appendix: bool = False, extra: bool = False) -> Path:
    base_dir = PAPER_EXTRA_DIR if extra else PAPER_TABLES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / _app_name(filename, appendix)


def morphalign_paper_score(value: float) -> float:
    return value * MORPHALIGN_SCORE_SCALE


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cache_key(corpus_name: str, method: str, bias: float) -> str:
    return f"{corpus_name}/{method}/bias{bias}"


def geomean(values):
    # MorphAlign across languages spans orders of magnitude (Finnish ~30x English),
    # so arithmetic mean is Finnish-dominated. Geometric mean weights each language
    # multiplicatively and is the natural aggregator on the log scale used in plots.
    vals = list(values)
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def apply_inference_bias(model: UnigramModel, bias: float):
    for token in model.tokens.values():
        token.log_prob -= bias


def remove_inference_bias(model: UnigramModel, bias: float):
    for token in model.tokens.values():
        token.log_prob += bias


def compute_biased_compression(model: UnigramModel, eval_corpus: str, bias: float) -> dict:
    apply_inference_bias(model, bias)
    corpus = load_corpus_by_name(eval_corpus, model.pretokenizer)
    perf = model.corpus_performance(corpus)
    remove_inference_bias(model, bias)
    return {"tokens": perf["total_tokens_len"], "bytes": perf["total_byte_len"]}


def escape_latex(text: str) -> str:
    return (
        text.replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def load_model_for_method(
    corpus_name: str,
    method: str,
    *,
    default_overshoot_factor: float = DEFAULT_OVERSHOOT_FACTOR,
    unigram_results_dir: Path = UNIGRAM_RESULTS_DIR,
):
    if method == "BPE":
        path = unigram_results_dir / corpus_name / f"bpe_n{ADDITIONAL_VOCAB_SIZE}.model.json.gz"
        if path.exists():
            return BPETokenizer.load(str(path)), path
        return None, None

    factor = default_overshoot_factor
    base_method = method
    if "/f=" in method:
        base_method, factor_str = method.split("/f=")
        factor = float(factor_str)

    if base_method in ["Default", "Default+Bias"]:
        path = get_unigram_model_path(corpus_name, DEFAULTS)
    elif base_method in ["FSP", "FSP+Bias"]:
        path = get_unigram_model_path(corpus_name, FSP_PARAMS)
    elif base_method in ["BPE-Init", "BPE-Init+Bias"]:
        path = get_hybrid_model_path(corpus_name, {**DEFAULTS, "overshoot_factor": factor})
    elif base_method in ["BPE-Init+FSP", "BPE-Init+FSP+Bias"]:
        path = get_hybrid_model_path(corpus_name, {**DEFAULTS, **FSP_OVERRIDES, "overshoot_factor": factor})
    else:
        raise ValueError(f"Unknown method: {method}")

    if path.exists():
        return UnigramModel.load(str(path)), path
    return None, None

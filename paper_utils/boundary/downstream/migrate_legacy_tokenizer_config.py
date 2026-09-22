"""Relabel the stored pretokenizer config of a 2026-08 boundary tokenizer to the current schema.

The tokenizers trained in August serialize a pretokenizer config that the current
`BoundaryScriptPretokenizerConfig` rejects:

    old                                          current
    cls: BoundaryScriptPretokenizer              same, for the no-case arms
    cls: ExtCapsFixBoundaryScriptPretokenizer    BoundaryScriptPretokenizer
    boundary_targets: ("word", "punct", ...)     ("punct", ...)      words are always delimited
    caps_codes: bool                             shift_code + caps_code

Only the labelling changed. The August `ExtCapsFix...` class and the current
`BoundaryScriptPretokenizer` with `shift_code=caps_code=True` are the same scheme: the case
code sits outside the span's markers and the case test is `istitle() or isupper()`. Same for
the no-case arms. The vocabulary in the file is untouched by this script.

Because "the same scheme" is an assertion about two implementations, this script does not
rely on it. For every file it compares the token ids produced by the August code reading the
original file against the token ids produced by current code reading the rewritten file, over
the same documents, and refuses to keep a rewrite that changes a single id.

    python -m paper_utils.boundary.downstream.migrate_legacy_tokenizer_config run \
        <tokenizer.json.gz> [<tokenizer.json.gz> ...] \
        --legacy-module <path to the August boundary_pretokenizer.py> \
        --workdir <scratch dir>

`--legacy-module` is the file as it exists on the branch that trained these tokenizers:

    git show fork/claude/mingram-five-arms:marker_experiments/boundary_pretokenizer.py > /tmp/old.py
"""
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cyclopts

# Documents encoded per file when comparing the two implementations. 200 documents of
# FineWeb-2 is 0.2M characters of Korean and 1.4M of Russian, which covers every span kind
# the schemes distinguish many times over.
COMPARE_DOCS = 200
FINEWEB2 = Path(
    "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_10-filterrobots/data/output"
)
LANG_DIR = {"ko": "kor_Hang", "ru": "rus_Cyrl"}
# FineWeb-2 has no English. English documents come from the held-out FineWiki slice the grid
# used for its own compression check, a JSON list of documents; gitignored, so its sha256 is
# printed with every comparison that reads it.
EN_SAMPLE = Path(__file__).resolve().parents[3] / "marker_experiments" / "eval_texts" / "en.json"
TRAINER_CLASS = {
    "bpe": ("script_bpe.tokenizers.bpe", "BPETokenizer"),
    "mingram": ("script_bpe.tokenizers.mingram.model", "MinGramModel"),
}

app = cyclopts.App()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["pretokenizer"]["config"]


def rewrite(path: Path, out: Path) -> tuple[dict, dict]:
    """Write `path` to `out` with only the pretokenizer config relabelled."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)

    old = dict(data["pretokenizer"]["config"])
    targets = tuple(old.get("boundary_targets", ()))
    if "word" not in targets:
        raise SystemExit(
            f"{path}: boundary_targets {targets} does not contain 'word'. This is not a "
            f"2026-08 boundary config; refusing to guess what it is."
        )
    if "caps_codes" not in old:
        raise SystemExit(f"{path}: no 'caps_codes' field. Not a 2026-08 boundary config.")

    legacy_cls = old["cls"]
    if legacy_cls not in ("BoundaryScriptPretokenizer", "ExtCapsFixBoundaryScriptPretokenizer"):
        raise SystemExit(
            f"{path}: pretokenizer class {legacy_cls!r}. Only the plain boundary class and "
            f"ExtCapsFix (case code outside the markers, case test requires case) are "
            f"relabelled here. ExtCaps without the fix codes every uncased word and is a "
            f"different scheme; it is not migrated."
        )
    caps = bool(old["caps_codes"])
    if caps and legacy_cls != "ExtCapsFixBoundaryScriptPretokenizer":
        raise SystemExit(
            f"{path}: caps codes with class {legacy_cls!r}, which places the code inside the "
            f"span's markers. Current code implements only the outside placement, so this "
            f"file has no current equivalent."
        )

    new = {k: v for k, v in old.items() if k != "caps_codes"}
    new["cls"] = "BoundaryScriptPretokenizer"
    new["boundary_targets"] = [t for t in targets if t != "word"]
    new["shift_code"] = caps
    new["caps_code"] = caps

    data["pretokenizer"]["config"] = new
    # Two places name the class: the envelope, which `load_pretokenizer` looks up in the
    # registry, and the config itself. Rewriting only the config leaves the loader hunting
    # for a class the current code does not define.
    data["pretokenizer"]["config_class"] = "BoundaryScriptPretokenizer"
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    return old, new


def sample_docs(lang: str, n: int) -> list[str]:
    if lang == "en":
        if not EN_SAMPLE.exists():
            raise SystemExit(f"no English sample at {EN_SAMPLE}")
        docs = json.loads(EN_SAMPLE.read_text())
        if len(docs) < n:
            raise SystemExit(f"{EN_SAMPLE} holds {len(docs)} documents, fewer than the {n} requested")
        print(f"[sample] en: {EN_SAMPLE} sha256 {sha256(EN_SAMPLE)[:16]}", file=sys.stderr)
        return docs[:n]

    import pyarrow.parquet as pq

    path = sorted((FINEWEB2 / LANG_DIR[lang]).glob("*.parquet"))[0]
    pf = pq.ParquetFile(path)
    docs: list[str] = []
    for rg in range(pf.metadata.num_row_groups):
        for s in pf.read_row_group(rg, columns=["text"]).column("text"):
            docs.append(s.as_py())
            if len(docs) >= n:
                return docs
    raise SystemExit(f"{path} holds {len(docs)} documents, fewer than the {n} requested")


@app.command
def encode(tokenizer: Path, trainer: str, lang: str, legacy_module: Path | None = None, out: Path = None):
    """Encode the sample documents and write one sha256 per document. Run as a subprocess.

    With `--legacy-module`, the August pretokenizer classes are registered and the current
    ones are never imported, so the file is read by the code that wrote it.
    """
    if legacy_module is not None:
        sys.path.insert(0, str(legacy_module.parent))
        __import__(legacy_module.stem)  # registers the August classes in Pretokenizer.REGISTRY
    else:
        __import__("paper_utils.boundary.boundary_pretokenizer")

    module, cls_name = TRAINER_CLASS[trainer]
    cls = getattr(__import__(module, fromlist=[cls_name]), cls_name)
    tok = cls.load(str(tokenizer))

    digests = []
    for doc in sample_docs(lang, COMPARE_DOCS):
        ids = tok.encode(doc)
        digests.append(hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest())
        if tok.decode(ids) != doc:
            raise SystemExit(f"{tokenizer}: round-trip failed on a sample document")
    out.write_text(json.dumps({"vocab": len(tok.tokens), "digests": digests}))


@app.command
def run(tokenizers: list[Path], legacy_module: Path, original_dir: Path, workdir: Path = Path("/tmp")):
    """Relabel each file, keeping it only if it encodes identically to the pristine original.

    `original_dir` holds the files as trained, under the same names as the targets. Every
    target is rewritten from its original rather than from whatever is in place, so a rerun
    is idempotent and a half-finished run leaves nothing to reason about.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    report = []
    for path in tokenizers:
        name = path.name
        original = original_dir / name
        if not original.exists():
            raise SystemExit(f"{name}: no pristine original at {original}")
        lang = next((code for code in ("ko", "ru", "en") if f"_{code}_" in name), None)
        trainer = "mingram" if "_mingram_" in name else "bpe"
        if lang is None:
            raise SystemExit(f"{name}: cannot tell the language from the file name")

        migrated = workdir / f"migrated_{name}"
        old_cfg, new_cfg = rewrite(original, migrated)

        old_out = workdir / f"{name}.old.json"
        new_out = workdir / f"{name}.new.json"
        base = [sys.executable, "-m", "paper_utils.boundary.downstream.migrate_legacy_tokenizer_config", "encode"]
        subprocess.run(
            base + [str(original), trainer, lang, "--legacy-module", str(legacy_module), "--out", str(old_out)],
            check=True,
        )
        subprocess.run(base + [str(migrated), trainer, lang, "--out", str(new_out)], check=True)

        old_res, new_res = json.loads(old_out.read_text()), json.loads(new_out.read_text())
        if old_res != new_res:
            differing = sum(a != b for a, b in zip(old_res["digests"], new_res["digests"]))
            raise SystemExit(
                f"{name}: the relabelled file does not encode identically to the original "
                f"({differing} of {COMPARE_DOCS} documents differ, vocab {old_res['vocab']} "
                f"vs {new_res['vocab']}). {path} is left as it was."
            )

        shutil.move(str(migrated), str(path))
        print(f"{name}\n  vocab {old_res['vocab']}, identical ids on {COMPARE_DOCS} {lang} documents"
              f"\n  sha256 {sha256(original)[:16]} -> {sha256(path)[:16]}"
              f"\n  {field_diff(old_cfg, new_cfg)}", flush=True)
        report.append(name)

    print(f"\n{len(report)} file(s) relabelled and verified")


def field_diff(old: dict, new: dict) -> str:
    keys = sorted(set(old) | set(new))
    return ", ".join(f"{k}: {old.get(k, '-')!r} -> {new.get(k, '-')!r}" for k in keys if old.get(k) != new.get(k))


if __name__ == "__main__":
    app()

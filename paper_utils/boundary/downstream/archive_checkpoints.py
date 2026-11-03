"""Copy finished runs' final model weights off scratch, to storage that is not purged.

capstor scratch deletes files not accessed for 14 days, and between 2026-08-04 and 2026-09-17
it deleted every checkpoint of the published English runs. This copies, per finished run,
the final `model_<step>.pt` and its `meta_<step>.json`, the run log, and the tokenizer the
run was trained with, into `<dest>/<tag>/`, and writes `archive.json` recording sha256 and
size of every file on both sides. Optimizer state is not copied: it is 1.6 times the size
of the model and only matters for resuming training.

A run counts as finished when its log holds the result block. Smoke runs are skipped. A copy
is written under a temporary name and renamed only after its sha256 matches the source, so
an interrupted copy never looks complete. Rerunning is safe: a run whose archive already
matches the source is left alone.

    python -m paper_utils.boundary.downstream.archive_checkpoints \
        --logs-dir results/marker_downstream_ko/logs --corpus fineweb_ko_5gb_quick \
        --dest /capstor/store/cscs/swissai/a0229/cmeister/script_tok_boundary/downstream_models/marker_downstream_ko
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cyclopts

from paper_utils.boundary.downstream.collect_results import TAG_RE

HERE = Path(__file__).resolve().parent
TOKENIZER_DIR = HERE / "tokenizers"

app = cyclopts.App()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def result_fields(log: Path) -> dict | None:
    """tokenizer_id and artifact_dir from the result block, or None if the run did not finish."""
    lines = log.read_text(errors="replace").splitlines()
    try:
        start = max(i for i, line in enumerate(lines) if line.strip() == "Downstream eval result")
    except ValueError:
        return None
    fields = {}
    for line in lines[start:]:
        key, _, value = line.strip().partition(":")
        if key.strip() in ("tokenizer_id", "artifact_dir"):
            fields[key.strip()] = value.strip()
    return fields if len(fields) == 2 else None


def final_checkpoint(checkpoint_dir: Path) -> tuple[Path, Path]:
    models = sorted(checkpoint_dir.glob("model_*.pt"))
    if not models:
        raise SystemExit(f"no model_*.pt under {checkpoint_dir}")
    model = models[-1]
    step = model.stem.removeprefix("model_")
    meta = checkpoint_dir / f"meta_{step}.json"
    if not meta.exists():
        raise SystemExit(f"{model} has no matching {meta.name}")
    return model, meta


def copy_verified(src: Path, dst: Path) -> dict:
    src_sha = sha256(src)
    if dst.exists() and sha256(dst) == src_sha:
        return {"file": dst.name, "sha256": src_sha, "bytes": src.stat().st_size, "source": str(src), "copied": False}
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp)
    if sha256(tmp) != src_sha:
        tmp.unlink()
        raise SystemExit(f"copy of {src} to {tmp} does not match its source")
    os.replace(tmp, dst)
    return {"file": dst.name, "sha256": src_sha, "bytes": src.stat().st_size, "source": str(src), "copied": True}


@app.default
def main(logs_dir: Path, corpus: str, dest: Path, tags: list[str] | None = None, vocab: int = 34685):
    """Archive every finished run under `logs_dir`, or only `tags` when given.

    Args:
        logs_dir: the sweep's logs directory, one `<tag>.log` per run.
        corpus: tokenizer corpus name, used to find the tokenizer file for each run.
        dest: archive root for this sweep; each run goes to `<dest>/<tag>/`.
        tags: restrict to these run tags. A job passes its own runs so that two jobs
            finishing together never copy the same run.
        vocab: total vocabulary in the tokenizer file names.
    """
    logs = sorted(logs_dir.glob("*.log"))
    if tags:
        wanted = set(tags)
        logs = [log for log in logs if log.stem in wanted]
        missing = wanted - {log.stem for log in logs}
        if missing:
            raise SystemExit(f"no log for tag(s) {sorted(missing)} under {logs_dir}")

    archived = 0
    for log in logs:
        fields = result_fields(log)
        if fields is None:
            print(f"[archive] {log.stem}: no result block, not finished, skipped")
            continue
        tag = fields["tokenizer_id"]
        if tag.endswith("_smoke"):
            continue
        m = TAG_RE.match(tag)
        if not m:
            raise SystemExit(f"{log}: tag {tag!r} does not parse as <arm>_<trainer>_d<depth>_s<seed>")

        model, meta = final_checkpoint(Path(fields["artifact_dir"]) / "base_checkpoints" / tag)
        tokenizer = TOKENIZER_DIR / f"{corpus}_{m['arm']}_{m['trainer']}_v{vocab}.json.gz"
        if not tokenizer.exists():
            raise SystemExit(f"{tag}: tokenizer {tokenizer} not found")

        out = dest / tag
        out.mkdir(parents=True, exist_ok=True)
        files = [copy_verified(src, out / src.name) for src in (model, meta, log, tokenizer)]
        record = {
            "tag": tag,
            "arm": m["arm"],
            "trainer": m["trainer"],
            "seed": int(m["seed"]),
            "corpus": corpus,
            "archived": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": files,
        }
        (out / "archive.json").write_text(json.dumps(record, indent=1))
        archived += 1
        gb = model.stat().st_size / 1e9
        print(f"[archive] {tag}: {model.name} ({gb:.2f} GB) and {len(files) - 1} other file(s) verified in {out}")

    print(f"[archive] {archived} run(s) archived under {dest}")


if __name__ == "__main__":
    app()

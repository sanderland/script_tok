"""Point a per-language nanochat base directory at FineWeb-2 text.

The downstream runner reads `shard_*.parquet` from `<NANOCHAT_BASE>/base_data_climbmix`,
and nanochat's loader takes every file but the last as training data and the last as
validation. That directory name is nanochat's, baked into its loader; for these runs it
holds FineWeb-2 in one language and no ClimbMix at all. `shard_provenance.json`, written
beside it, records what each shard really is, because the directory name does not.

Each shard is a symlink to a file in the shared read-only FineWeb-2 copy. Nothing is
duplicated, and the source files carry a `text` column, which is what the loader reads.

How many training shards a language gets is set by matching the number of passes over the
training text that the English runs made, about 3.5. Steps per run are fixed by the model,
so the tokens a run consumes are fixed too, and the passes over the text follow from how
much text there is and how many tokens the tokenizer makes of it. Korean text yields about
twice as many tokens per character as English, so the same number of passes needs less text.

    python -m paper_utils.boundary.downstream.build_language_shards \
        --lang ko --train-shards 3 --base-dir /capstor/scratch/cscs/$USER/marker_downstream/nanochat_base_ko
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import cyclopts

FINEWEB2 = Path(
    "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_10-filterrobots/data/output"
)
LANG_DIR = {"ko": "kor_Hang", "ru": "rus_Cyrl"}
# nanochat's own directory name, not a description of the contents. See the module docstring.
DATA_SUBDIR = "base_data_climbmix"

app = cyclopts.App()


@app.default
def build(
    lang: str,
    base_dir: Path,
    train_shards: int,
    source_dir: Path | None = None,
    measured_chars: Path | None = None,
):
    """Create `train_shards` training shards plus one validation shard, as symlinks.

    Args:
        lang: ko or ru.
        base_dir: the NANOCHAT_BASE for this language. Must not already hold shards.
        train_shards: how many training shards to link. One more file is linked for
            validation, which nanochat takes from the end of the sorted list.
        source_dir: override the FineWeb-2 language directory.
        measured_chars: JSON from the character count, recorded in the provenance file.
    """
    if lang not in LANG_DIR:
        raise SystemExit(f"unknown language {lang!r}; have {sorted(LANG_DIR)}")
    src = source_dir or (FINEWEB2 / LANG_DIR[lang])
    files = sorted(src.glob("*.parquet"))
    need = train_shards + 1
    if len(files) < need:
        raise SystemExit(f"{src} holds {len(files)} parquet files, fewer than the {need} needed")

    data_dir = base_dir / DATA_SUBDIR
    existing = sorted(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    if existing:
        raise SystemExit(
            f"{data_dir} already holds {len(existing)} shard(s). Refusing to change the data "
            f"under a directory that runs may already have trained against. Remove it "
            f"explicitly if that is what you want."
        )
    data_dir.mkdir(parents=True, exist_ok=True)

    chars = {}
    if measured_chars is not None:
        chars = {e["file"]: e["chars"] for e in json.loads(measured_chars.read_text()).get(lang, [])}

    shards = []
    for i, source in enumerate(files[:need]):
        link = data_dir / f"shard_{i:05d}.parquet"
        link.symlink_to(source)
        shards.append(
            {
                "shard": link.name,
                "role": "validation" if i == need - 1 else "train",
                "source": str(source),
                "bytes": source.stat().st_size,
                "characters": chars.get(source.name),
            }
        )

    train_chars = sum(s["characters"] for s in shards if s["role"] == "train" and s["characters"])
    provenance = {
        "language": lang,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": str(src),
        "note": (
            f"{DATA_SUBDIR} is nanochat's directory name and holds no ClimbMix here: these "
            f"shards are FineWeb-2 {lang}. The last shard is validation, the rest training."
        ),
        "train_shards": train_shards,
        "train_characters": train_chars or None,
        "shards": shards,
    }
    (base_dir / "shard_provenance.json").write_text(json.dumps(provenance, indent=1))

    print(f"{lang}: {train_shards} training shards + 1 validation shard under {data_dir}")
    if train_chars:
        print(f"  training text: {train_chars:,} characters")
    print(f"  provenance: {base_dir / 'shard_provenance.json'}")


@app.command
def rehash(base_dir: Path):
    """Add a sha256 for every shard to an existing provenance file.

    `build` links shards and records names, sizes and character counts. A reader who cannot
    see this filesystem needs the hashes as well, to tell whether a file they hold is the
    same one. This fills them in without touching the selection or anything else recorded.
    """
    import hashlib

    out = base_dir / "shard_provenance.json"
    provenance = json.loads(out.read_text())
    for shard in provenance["shards"]:
        path = (base_dir / DATA_SUBDIR / shard["shard"]).resolve()
        if path.stat().st_size != shard["bytes"]:
            raise SystemExit(f"{path} is {path.stat().st_size} bytes, provenance says {shard['bytes']}")
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                digest.update(block)
        shard["sha256"] = digest.hexdigest()
        print(f"  {shard['shard']} {shard['sha256'][:16]} {path}")
    provenance["hashed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.write_text(json.dumps(provenance, indent=1))
    print(f"{len(provenance['shards'])} shard(s) hashed in {out}")


@app.command
def record(lang: str, base_dir: Path, source: str):
    """Write shard_provenance.json for shards something else put in place.

    For the English retrain, whose shards are ClimbMix downloaded by nanochat's own
    `python -m nanochat.dataset -n 8`, not links into FineWeb-2. The launcher refuses a base
    directory without this file, so the language check covers English too. Each shard's
    sha256 is recorded, since a download is only as reproducible as its content.

    Args:
        lang: language of the text, as it appears in the tokenizer corpus name.
        base_dir: the NANOCHAT_BASE holding base_data_climbmix/.
        source: free-text description of where the shards came from.
    """
    import hashlib

    data_dir = base_dir / DATA_SUBDIR
    files = sorted(data_dir.glob("shard_*.parquet"))
    if len(files) < 2:
        raise SystemExit(f"{data_dir} holds {len(files)} shard(s); need training shards and a validation shard")
    out = base_dir / "shard_provenance.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite a provenance record")

    shards = []
    for i, f in enumerate(files):
        digest = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                digest.update(block)
        shards.append(
            {
                "shard": f.name,
                "role": "validation" if i == len(files) - 1 else "train",
                "source": str(f.resolve()),
                "bytes": f.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    provenance = {
        "language": lang,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": source,
        "note": "The last shard is validation, the rest training.",
        "train_shards": len(files) - 1,
        "shards": shards,
    }
    out.write_text(json.dumps(provenance, indent=1))
    print(f"{lang}: {len(files) - 1} training shards + 1 validation shard recorded in {out}")


if __name__ == "__main__":
    app()

"""Verify and export the four 80/20 MinGram models for downstream use."""

import hashlib
import json
from pathlib import Path
import shutil

from paper_utils.boundary.downstream.boundary_tokenizer import BoundaryMinGramModel

ROOT = Path(__file__).resolve().parents[2]
CORPUS = "fineweb_enko_5000m_en80_local_v2"
DEST = ROOT / "paper_utils/boundary/downstream/tokenizers/en80_ko20_mingram_64k"
ARMS = {"plain": 65710, "bnd_w": 65711, "bnd_wpd": 65711, "bnd_wpd_caps": 65713}


def main():
    verified = []
    sample = evaluation = None
    for arm, size in ARMS.items():
        source = ROOT / f"results/boundary_bilingual/{CORPUS}_v{size}_mingram"
        info = json.loads((source / f"{arm}.train.json").read_text())
        analysis = json.loads((source / f"{arm}.analysis.json").read_text())
        model_path = source / f"{arm}.json.gz"
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert digest == info["tokenizer_sha256"]
        assert info["trainer"] == "mingram" and info["corpus"] == CORPUS
        assert info["additional_vocab_size"] == 64000
        assert analysis["analysis"]["vocabulary"]["learned_tokens"] == 64000
        if sample is None:
            sample, evaluation = info["sample"], analysis["evaluation"]
        assert sample == info["sample"] and evaluation == analysis["evaluation"]
        assert all(v["roundtrip_failures"] == 0 for v in analysis["analysis"]["evaluation"].values())
        model = BoundaryMinGramModel.load(str(model_path))
        assert len(model.tokens) == size
        for text in ["Hello world! 123", "한국어와 English 123."]:
            assert model.decode(model.encode(text)) == text
        verified.append((arm, source, digest, size))
    DEST.mkdir(parents=True, exist_ok=True)
    entries = []
    for arm, source, digest, size in verified:
        for suffix in ("json.gz", "train.json", "analysis.json"):
            shutil.copyfile(source / f"{arm}.{suffix}", DEST / f"{arm}.{suffix}")
        assert hashlib.sha256((DEST / f"{arm}.json.gz").read_bytes()).hexdigest() == digest
        entries.append(dict(arm=arm, path=f"{arm}.json.gz", sha256=digest,
                            total_tokens=size, learned_tokens=64000,
                            training=f"{arm}.train.json", analysis=f"{arm}.analysis.json"))
    manifest = dict(corpus=CORPUS, training_characters={"en": 4000000000, "ko": 1000000000},
                    tokenizer_class="paper_utils.boundary.downstream.boundary_tokenizer.BoundaryMinGramModel",
                    path_base="manifest directory", tokenizers=entries)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified and packaged {len(entries)} tokenizers at {DEST}")


if __name__ == "__main__":
    main()

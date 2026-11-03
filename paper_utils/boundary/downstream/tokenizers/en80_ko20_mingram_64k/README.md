# English–Korean downstream tokenizers

Four MinGram models trained on the same 5 billion characters: 80% English,
20% Korean. Each has 64,000 learned tokens plus fixed atomics. The schemes are
plain, boundary[w], boundary[w,p,d], and boundary[w,p,d,↑], corresponding to
`plain`, `bnd_w`, `bnd_wpd`, and `bnd_wpd_caps`.

Only this README and `manifest.json` are tracked; the model files are not (see
`../.gitignore`). `manifest.json` contains relative model paths, SHA256 hashes and total
vocabulary sizes, so a regenerated or downloaded copy can be checked against it. Resolve
paths relative to this directory. Each model includes its training
provenance and intrinsic evaluation. Absolute source-cache paths in provenance
record the original machine; loading the models does not require those caches.

From the repository root, with project dependencies installed:

```python
from paper_utils.boundary.downstream.boundary_tokenizer import BoundaryMinGramModel

model = BoundaryMinGramModel.load(
    "paper_utils/boundary/downstream/tokenizers/en80_ko20_mingram_64k/bnd_w.json.gz"
)
ids = model.encode("Hello 한국어!")
assert model.decode(ids) == "Hello 한국어!"
```

Use the manifest's total vocabulary size for downstream embeddings. Preserve the
serialized token IDs. The ↑ scheme has two more fixed atomics than the other
boundary schemes. These artifacts contain tokenizer results only; downstream LM
evaluation remains to be run.

To reproduce the export after training, run
`python -m paper_utils.boundary.package_bilingual`.

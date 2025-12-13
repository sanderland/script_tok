from script_bpe.train import tokenizer_save_path, train_tokenizer


def test_train_smoke_prepares_only(tmp_path):
    # additional_vocab_size=0 skips training; use swift corpus to avoid HF
    tok = train_tokenizer(
        pretokenizer_name="scriptenc_cb",
        model_name="bpe",
        corpus_name="swift",
        additional_vocab_size=0,
        n_cpus=1,
        retrain=True,
        report=False,
    )
    assert tok is None

    # path helper yields a sensible path string
    path = tokenizer_save_path("swift", 10, "scriptenc_cb", "bpe")
    assert path.endswith("swift/n10/scriptenc_cb.json.gz")

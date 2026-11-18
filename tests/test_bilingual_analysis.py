from dataclasses import dataclass

from paper_utils.boundary import bilingual_analysis


@dataclass
class FakeToken:
    id: int
    atomic_tokens: list[int]


class FakePretokenizer:
    strings = {1: "a", 2: "b", 3: " ", 4: "A", 5: "가", 6: "나", 7: "Ж", 8: "1", 9: "2"}

    def normalize(self, text):
        return text

    def try_decode_strict(self, ids):
        if 88 in ids or 99 in ids:
            return None
        return "".join(self.strings[token_id] for token_id in ids)

    def decode(self, ids, errors="replace"):
        return "".join(self.strings.get(token_id, "") for token_id in ids)


class FakeTokenizer:
    def __init__(self):
        self.pretokenizer = FakePretokenizer()
        self.tokens = {
            0: FakeToken(0, [1]),  # Atomic tokens do not enter vocabulary statistics.
            10: FakeToken(10, [1, 2]),
            11: FakeToken(11, [3, 1, 2]),
            12: FakeToken(12, [4, 2]),
            13: FakeToken(13, [3, 4, 2]),
            14: FakeToken(14, [5, 6]),
            15: FakeToken(15, [1, 5]),
            16: FakeToken(16, [7, 7]),
            17: FakeToken(17, [8, 9]),
            18: FakeToken(18, [99, 1, 2, 99]),
            19: FakeToken(19, [88, 1]),
        }
        self.encoded = {"ab Ab": [10, 13], "가나": [14]}

    def encode(self, text):
        return self.encoded[text]

    def decode(self, ids):
        atomic = [item for token_id in ids for item in self.tokens[token_id].atomic_tokens]
        return self.pretokenizer.decode(atomic)


def test_script_properties_include_letters_without_script_in_unicode_name():
    assert bilingual_analysis._classify_text("K") == "latin"
    assert bilingual_analysis._classify_text("e\u0301") == "latin"
    assert bilingual_analysis._classify_text("ㄱ한") == "hangul"
    assert bilingual_analysis._classify_text("a한") == "mixed"


def test_analyze_tokenizer_classifies_vocab_and_duplicate_surplus(monkeypatch):
    # The fake marker cannot be an actual BoundaryScriptPretokenizer, but it exercises
    # the same contract: markers are visible in duplicate surfaces and absent from script
    # classification and strict decoding.
    monkeypatch.setattr(bilingual_analysis, "special_texts", lambda _pre: {99: "<M>"})

    result = bilingual_analysis.analyze_tokenizer(FakeTokenizer(), {"en": ["ab Ab"], "ko": ["가나"]})

    assert result["vocabulary"]["categories"] == {
        "latin": 5,
        "hangul": 1,
        "mixed": 1,
        "other_script": 1,
        "nonlexical": 1,
        "undecodable": 1,
    }
    duplicates = result["vocabulary"]["duplicates"]
    assert duplicates["space"]["involved_entries"] == 4
    assert duplicates["space"]["surplus_entries"] == 2
    assert duplicates["case"]["involved_entries"] == 4
    assert duplicates["case"]["surplus_entries"] == 2
    assert duplicates["combined"]["involved_entries"] == 4
    assert duplicates["combined"]["surplus_entries"] == 3
    assert duplicates["combined"]["by_script"]["latin"] == {"involved_entries": 4, "surplus_entries": 3}
    assert sum(result["vocabulary"]["categories"].values()) == result["vocabulary"]["learned_tokens"]
    assert sum(row["involved_entries"] for row in duplicates["combined"]["by_script"].values()) == 4
    assert sum(row["surplus_entries"] for row in duplicates["combined"]["by_script"].values()) == 3


def test_analyze_tokenizer_reports_streamed_evaluation_metrics(monkeypatch):
    monkeypatch.setattr(bilingual_analysis, "special_texts", lambda _pre: {99: "<M>"})

    result = bilingual_analysis.analyze_tokenizer(FakeTokenizer(), {"en": ["ab Ab"], "ko": ["가나"]})

    assert result["evaluation"]["en"] == {
        "documents": 1,
        "characters": 5,
        "bytes": 5,
        "tokens": 2,
        "chars_per_token": 2.5,
        "bytes_per_token": 2.5,
        "roundtrip_failures": 0,
        "script_usage": {"latin": 2, "hangul": 0, "mixed": 0, "other_script": 0, "nonlexical": 0, "undecodable": 0},
        "unique_token_counts": {"latin": 2, "hangul": 0, "mixed": 0, "other_script": 0, "nonlexical": 0, "undecodable": 0},
    }
    assert result["evaluation"]["ko"]["bytes_per_token"] == 6.0
    assert result["evaluation"]["ko"]["script_usage"]["hangul"] == 1
    assert sum(result["evaluation"]["ko"]["script_usage"].values()) == result["evaluation"]["ko"]["tokens"]

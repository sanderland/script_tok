import pytest

from script_bpe.pretokenize import get_pretokenizer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello world", ["hello", " world"]),
        (" leading", [" leading"]),
        ("a  b", ["a", " ", " b"]),
        ("a   b  ", ["a", " ", " ", " b", " ", " "]),
        ("  ", [" ", " "]),
        ("tab\tsep\nline", ["tab\tsep\nline"]),
        ("a \n b", ["a", " \n", " b"]),
        ("你好 世界,是的", ["你好", " 世界,是的"]),
        ("abc123 4567.8", ["abc123", " 4567.8"]),
        ("x▁y z", ["x▁y", " z"]),
    ],
)
def test_sp_split_rule(text, expected):
    pt = get_pretokenizer("bytes_sp_cb")
    chunks = pt.pretokenize(text)
    assert [pt.decode(c) for c in chunks] == expected

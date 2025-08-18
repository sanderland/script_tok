import pytest

from script_bpe.pretokenize import PRETOKENIZER_REGISTRY


@pytest.mark.parametrize("pretokenizer_name", list(PRETOKENIZER_REGISTRY.keys()))
def test_pretokenizers_in_registry(pretokenizer_name):
    sample_strings = [
        "Hello world",
        "1234 مَبْنِي",
        "世 界!",
        "한글o ❤️",
        "a\nb\nc\r\nd",
        "❤️\u200cInvisible Zero Width Non-Joiner ",
        "~\u202eRight-To-Left Override",
        " \u000dCarriage Return",
    ]

    pretokenizer = PRETOKENIZER_REGISTRY[pretokenizer_name]()
    for s in sample_strings:
        # Encode and chunk the string
        encoded_chunks = pretokenizer.encode_and_chunk(s)
        # Decode the chunks back to a string
        decoded_string = pretokenizer.decode([item for sublist in encoded_chunks for item in sublist])
        # Ensure round-trip consistency
        assert decoded_string == pretokenizer.normalize(s), (
            f"Failed for pretokenizer: {pretokenizer_name} on string: {s!r}"
        )

    # test normalization does not remove unicode 16.0 introduced characters
    assert pretokenizer.normalize("1 \U000142aa ") == "1 \U000142aa "
    # Cn: Not Assigned (U+0378)  Co: Private Use (U+E000)   Cs: Surrogate (U+D800)
    assert pretokenizer.normalize(":\u0378\ue000)\ud800") == ":)"

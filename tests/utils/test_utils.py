from script_bpe.utils import create_logger, token_array


def test_token_array_and_logger():
    arr = token_array([1, 2, 3])
    assert arr.tolist() == [1, 2, 3]

    logger = create_logger("test", verbose=False)
    logger.info("hello")
    assert logger.name == "test"

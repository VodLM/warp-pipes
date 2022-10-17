import pytest

from warp_pipes.pipes.passages import gen_passages


@pytest.mark.parametrize(
    "cfg",
    [
        {
            "seq_length": 100,
            "size": 10,
            "stride": 7,
            "pad_token": None,
        },
        {
            "seq_length": 99,
            "size": 10,
            "stride": 7,
            "pad_token": None,
        },
        {
            "seq_length": 100,
            "size": 10,
            "stride": 10,
            "pad_token": None,
        },
        {
            "seq_length": 100,
            "size": 10,
            "stride": 8,
            "pad_token": None,
        },
        {
            "seq_length": 100,
            "size": 10,
            "stride": 10,
            "pad_token": -1,
        },
        {
            "seq_length": 100,
            "size": 10,
            "stride": 8,
            "pad_token": "[PAD]",
        },
        {
            "seq_length": 100,
            "size": 10,
            "stride": 10,
            "pad_token": -1,
            "start_tokens": ["<cls>"],
        },
        {
            "seq_length": 100,
            "size": 10,
            "stride": 8,
            "pad_token": "[PAD]",
            "start_tokens": [1, 2],
        },
    ],
)
def test_gen_passages(cfg, verbose=False):
    seq_length = cfg.pop("seq_length")
    x = list(range(seq_length))
    if verbose:
        print(f"\n> Input: {x}")
    tokens = []
    outputs = []
    for w, m in gen_passages(x, **cfg):
        outputs += [(w, m)]
        if verbose:
            print([ww if mm else "*" for ww, mm in zip(w, m)], len(w), len(m))
        tokens += [ww for ww, mm in zip(w, m) if mm > 0]

    # test that all input tokens (x) are placed once and only once in the windows
    assert all([xx == tt for xx, tt in zip(x, tokens)])

    # test that all windows and masks are of the same size
    assert all(len(w) == len(m) for w, m in outputs)

    # test that all sequences are of the same size if a pad_token is provided
    if cfg["pad_token"] is not None:
        assert all(len(w) == cfg["size"] for w, m in outputs)
        assert all(len(m) == cfg["size"] for w, m in outputs)

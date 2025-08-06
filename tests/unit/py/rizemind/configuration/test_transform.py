from datetime import datetime

import pytest
from rizemind.configuration.transform import (
    flatten,
    from_config,
    normalize,
    to_config,
    to_config_record,
    unflatten,
)

###############################################################################
# normalize
###############################################################################


def test_normalize_scalars_are_kept():
    """Scalars should remain unchanged."""
    src = {"i": 1, "f": 3.14, "s": "txt", "b": b"bytes", "flag": False}
    assert normalize(src) == src


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3],  # ints
        [3.14, 2.71],  # floats
        ["a", "b"],  # strs
        [b"x", b"y"],  # bytes
        [True, False, True],  # bools
    ],
)
def test_normalize_homogeneous_lists(values):
    src = {"key": values}
    assert normalize(src)["key"] == values


def test_normalize_mixed_list_raises():
    with pytest.raises(TypeError):
        normalize({"bad": [1, "two"]})


def test_normalize_nested_dict_and_none_filtered():
    src = {
        "lvl1": {"lvl2": {"val": 42}},
        "keep": "yes",
        "drop": None,  # should be removed
    }
    expected = {"lvl1": {"lvl2": {"val": 42}}, "keep": "yes"}
    assert normalize(src) == expected


def test_normalize_other_types_become_strings():
    now = datetime(2025, 1, 1, 12, 0, 0)
    res = normalize({"ts": now})
    assert res["ts"] == str(now)


###############################################################################
# to_config_record
###############################################################################


def test_to_config_record_wraps_normalize():
    """Ensures the helper delegates to normalize and wraps the result."""
    from flwr.common.record.configrecord import ConfigRecord

    data = {"a": 123, "b": [1, 2]}
    normalized = normalize(data)
    expected = ConfigRecord(normalized)
    assert to_config_record(data) == expected


###############################################################################
# flatten
###############################################################################


def test_flatten_simple():
    assert flatten({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_flatten_nested():
    src = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    expected = {"a.b": 1, "a.c.d": 2, "e": 3}
    assert flatten(src) == expected


def test_flatten_with_prefix():
    src = {"k": 7}
    expected = {"pref.k": 7}
    assert flatten(src, prefix="pref") == expected


###############################################################################
# to_config, unflatten, from_config  (round-trip checks)
###############################################################################


def test_to_config_round_trip():
    raw = {
        "section": {"item": [10, 20]},
        "flag": True,
        "omit": None,  # gets dropped by normalize
    }

    cfg = to_config(raw)
    # expected flattening (without "omit")
    assert cfg == {"section.item": [10, 20], "flag": True}

    # round-trip through from_config brings us back
    # to the **normalized** version (None removed, lists preserved)
    rebuilt = from_config(cfg)
    assert rebuilt == normalize({k: v for k, v in raw.items() if k != "omit"})


def test_unflatten_standalone():
    flat = {"p.q": 1, "p.r.s": "x", "t": False}
    expected = {"p": {"q": 1, "r": {"s": "x"}}, "t": False}
    assert unflatten(flat) == expected

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bioengbench.core.seeds import SEED_BITS, from_hex, new_seed, read_seed_file, streams, to_hex


@given(st.integers(min_value=0, max_value=2**SEED_BITS - 1))
def test_hex_round_trip(seed: int) -> None:
    assert from_hex(to_hex(seed)) == seed
    assert len(to_hex(seed)) == 32


def test_new_seeds_are_full_width_and_distinct() -> None:
    seeds = {new_seed() for _ in range(200)}
    assert len(seeds) == 200
    assert max(seeds).bit_length() > SEED_BITS - 8  # overwhelmingly likely


@pytest.mark.parametrize("bad", ["", "abc", "g" * 32, "0" * 31, "0" * 33])
def test_malformed_hex_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        from_hex(bad)


def test_seed_file_ignores_comments(tmp_path) -> None:
    f = tmp_path / "s.txt"
    f.write_text(f"# header\n\n{to_hex(5)}  # five\n{to_hex(7)}\n")
    assert read_seed_file(f) == [5, 7]


def test_streams_are_reproducible_and_independent() -> None:
    a1, b1 = streams(123, 2)
    a2, b2 = streams(123, 2)
    x1, y1 = a1.normal(size=1000), b1.normal(size=1000)
    assert np.array_equal(x1, a2.normal(size=1000))
    assert np.array_equal(y1, b2.normal(size=1000))
    assert abs(np.corrcoef(x1, y1)[0, 1]) < 0.15

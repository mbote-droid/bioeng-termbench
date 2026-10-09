"""Hidden-case seeds.

Seeds are 128-bit integers written as 32 hexadecimal characters. At 2^128
possible values, recovering a hidden case by trying seeds until the generated
noisy signal matches the input is infeasible, even with the generator in hand.
(The previous design used small integers, which an agent could brute-force.)

A seed is expanded with numpy's SeedSequence, so independent random streams
for one case (beat timing, noise, ...) come from `spawn`, never from
`seed + constant` arithmetic, which can make streams of different cases overlap.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np

SEED_BITS = 128
SEED_HEX_LEN = SEED_BITS // 4


def new_seed() -> int:
    """Draw a fresh seed from the operating system's CSPRNG."""
    return secrets.randbits(SEED_BITS)


def to_hex(seed: int) -> str:
    if not 0 <= seed < 2**SEED_BITS:
        raise ValueError(f"seed must be a {SEED_BITS}-bit non-negative integer")
    return f"{seed:0{SEED_HEX_LEN}x}"


def from_hex(text: str) -> int:
    text = text.strip().lower()
    if len(text) != SEED_HEX_LEN:
        raise ValueError(f"seed must be {SEED_HEX_LEN} hex characters, got {len(text)}")
    return int(text, 16)


def read_seed_file(path: Path) -> list[int]:
    """One hex seed per line; blank lines and '#' comments are ignored."""
    seeds = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            seeds.append(from_hex(line))
    if not seeds:
        raise ValueError(f"no seeds found in {path}")
    return seeds


def streams(seed: int, n: int) -> list[np.random.Generator]:
    """`n` statistically independent generators derived from one seed."""
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(n)]

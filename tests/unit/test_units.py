from bioengbench.checks.units import Tolerance, quantity


def test_absolute_floor_and_relative_band() -> None:
    tol = Tolerance(quantity(3.0, "ms"), 0.10)
    assert tol.within(quantity(22.9, "ms"), quantity(20.0, "ms"))  # 10 % = 2 ms < 3 ms floor
    assert not tol.within(quantity(23.1, "ms"), quantity(20.0, "ms"))
    assert tol.within(quantity(109.0, "ms"), quantity(100.0, "ms"))  # 10 % = 10 ms > floor
    assert not tol.within(quantity(110.5, "ms"), quantity(100.0, "ms"))


def test_unit_conversion_is_explicit() -> None:
    tol = Tolerance(quantity(3.0, "ms"))
    assert tol.within(quantity(0.0501, "s"), quantity(50.0, "ms"))
    assert not tol.within(quantity(0.050, "ms"), quantity(50.0, "ms"))  # the classic slip


def test_heart_rate_units() -> None:
    tol = Tolerance(quantity(1.0, "beat/minute"))
    hr = quantity(72.0, "beat/minute")
    assert tol.within(quantity(1.2, "beat/second"), hr)  # 72 bpm exactly
    assert not tol.within(quantity(72.0 / 60.0, "beat/minute"), hr)  # reported in Hz

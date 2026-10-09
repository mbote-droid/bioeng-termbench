import json

from bioengbench.core.scoring import CaseResult, CheckResult, Status, Verdict


def _verdict(statuses: list[Status]) -> Verdict:
    results = tuple(CheckResult(f"C{i}", s, measured=0.1 * i) for i, s in enumerate(statuses))
    return Verdict("t", "1.0.0", (CaseResult(1, results),))


def test_reward_is_binary_and_fraction_is_diagnostic(tmp_path) -> None:
    v = _verdict([Status.PASS, Status.FAIL, Status.PASS, Status.SKIPPED])
    reward = v.reward()
    assert reward["reward"] == 0.0
    assert reward["check_pass_fraction"] == 0.5
    v.write_reward(tmp_path)
    assert (tmp_path / "reward.txt").read_text() == "0\n"
    assert json.loads((tmp_path / "reward.json").read_text())["reward"] == 0.0


def test_all_pass_gives_reward_one() -> None:
    assert _verdict([Status.PASS, Status.PASS]).reward()["reward"] == 1.0


def test_empty_verdict_never_passes() -> None:
    assert not Verdict("t", "1").passed
    assert not CaseResult(1, ()).passed


def test_json_is_deterministic() -> None:
    a = _verdict([Status.PASS, Status.FAIL]).to_json()
    b = _verdict([Status.PASS, Status.FAIL]).to_json()
    assert a == b
    assert json.loads(a)["cases"][0]["seed"] == "0" * 31 + "1"

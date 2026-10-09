#!/usr/bin/env bash
# Verifier entry point. The agent's /app/pkmodel package directory is copied
# in as an artifact; the grader runs `python -m pkmodel.cli` as the
# unprivileged `sandbox` user with the package on PYTHONPATH.
set -euo pipefail

readonly TASK="t05-pk-debug"
readonly SUBMISSION="/app/pkmodel"
readonly REWARD_DIR="/logs/verifier"

seeds="/tests/dev_seeds.txt"
if [[ -s /tests/private_seeds.txt ]]; then
  seeds="/tests/private_seeds.txt"
fi

mkdir -p "${REWARD_DIR}"
echo 0 > "${REWARD_DIR}/reward.txt"  # fail closed if anything below crashes

if [[ ! -f "${SUBMISSION}/cli.py" ]]; then
  echo "no package at ${SUBMISSION}" >&2
  exit 0
fi

/opt/grader/bin/bioeng grade "${TASK}" \
  --submission "${SUBMISSION}" \
  --seeds "${seeds}" \
  --runner local \
  --sandbox-user sandbox \
  --python /usr/local/bin/python \
  --reward-dir "${REWARD_DIR}" || true

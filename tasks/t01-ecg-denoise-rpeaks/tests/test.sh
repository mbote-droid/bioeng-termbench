#!/usr/bin/env bash
# Verifier entry point. Runs as root in a fresh verifier container; the agent's
# /app/pipeline.py is copied in as an artifact. The submission itself runs as
# the unprivileged `sandbox` user, which cannot read /tests (seeds, grader).
set -euo pipefail

readonly TASK="t01-ecg-denoise-rpeaks"
readonly SUBMISSION="/app/pipeline.py"
readonly REWARD_DIR="/logs/verifier"

seeds="/tests/dev_seeds.txt"
if [[ -s /tests/private_seeds.txt ]]; then
  seeds="/tests/private_seeds.txt"
fi

mkdir -p "${REWARD_DIR}"
echo 0 > "${REWARD_DIR}/reward.txt"  # fail closed if anything below crashes

if [[ ! -f "${SUBMISSION}" ]]; then
  echo "no submission at ${SUBMISSION}" >&2
  exit 0
fi

# Exit status 1 means "graded and failed"; the reward files record that.
/opt/grader/bin/bioeng grade "${TASK}" \
  --submission "${SUBMISSION}" \
  --seeds "${seeds}" \
  --runner local \
  --sandbox-user sandbox \
  --python /usr/local/bin/python \
  --reward-dir "${REWARD_DIR}" || true

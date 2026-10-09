#!/usr/bin/env bash
# Replace the four defective modules with the fixed ones and prove the fix.
set -euo pipefail
cp /solution/pkmodel/{units,grid,regimen,solver}.py /app/pkmodel/
cp /solution/pkmodel/tests/test_pkmodel.py /app/pkmodel/tests/
cd /app
python -m pytest -q pkmodel/tests
python -m pkmodel.cli --params data/bolus/params.json --regimen data/bolus/regimen.json --out /tmp/conc.csv
head -3 /tmp/conc.csv

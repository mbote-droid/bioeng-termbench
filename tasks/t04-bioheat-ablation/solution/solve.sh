#!/usr/bin/env bash
set -euo pipefail
cp /solution/ablation.py /app/ablation.py
python /app/ablation.py plan --problem /app/data/problem.json --outdir /app/out
cat /app/out/plan.json

#!/usr/bin/env bash
# Oracle: install the reference pipeline where the instruction asks for it.
set -euo pipefail

cp /solution/pipeline.py /app/pipeline.py
python /app/pipeline.py --input /app/data/ecg_raw.csv --fs 360 --outdir /app/out

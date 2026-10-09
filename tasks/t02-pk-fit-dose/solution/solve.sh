#!/usr/bin/env bash
set -euo pipefail
cp /solution/pk_pipeline.py /app/pk_pipeline.py
python /app/pk_pipeline.py --indir /app/data --outdir /app/out

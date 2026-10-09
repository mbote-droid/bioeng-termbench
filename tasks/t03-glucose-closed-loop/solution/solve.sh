#!/usr/bin/env bash
set -euo pipefail
cp /solution/controller.py /app/controller.py
python /app/data/simulate.py /app/controller.py --patients 5

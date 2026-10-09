# Base image for every task's agent environment.
# Contains Python and the pinned numerical stack only: no grader, no generator.
# Build from the repository root:
#   docker build -f docker/agent-base.Dockerfile -t ghcr.io/mbote-droid/bioengbench-agent-base:0.2.0 .
# scripts/pin_base_images.sh replaces the tag below with an immutable digest.
FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY docker/agent-requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/agent-requirements.txt \
 && rm /tmp/agent-requirements.txt

WORKDIR /app

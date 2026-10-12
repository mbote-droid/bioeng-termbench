# Base image for every task's verifier.
#
# Two separate Python environments, on purpose:
#   /usr/local/bin/python   numpy + scipy, identical to the agent image; runs the
#                           submission as the unprivileged `sandbox` user.
#   /opt/grader             root-only (mode 0700) venv with bioengbench, pint and
#                           the generators; runs the grading logic.
# The submission therefore cannot import the generator or read seeds in /tests.
#
# Build from the repository root:
#   docker build -f docker/grader-base.Dockerfile -t ghcr.io/mbote-droid/bioengbench-grader-base:0.2.0 .
FROM python:3.12.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY docker/agent-requirements.txt /tmp/agent-requirements.txt
COPY docker/grader-requirements.txt /tmp/grader-requirements.txt
COPY dist/ /tmp/dist/

RUN pip install --no-cache-dir --require-hashes -r /tmp/agent-requirements.txt \
 && python -m venv /opt/grader \
 && /opt/grader/bin/pip install --no-cache-dir --require-hashes -r /tmp/grader-requirements.txt \
 && /opt/grader/bin/pip install --no-cache-dir --no-deps /tmp/dist/*.whl \
 && rm -rf /tmp/*requirements.txt /tmp/dist \
 && useradd --system --no-create-home --shell /usr/sbin/nologin sandbox \
 && chmod 0700 /opt/grader

WORKDIR /app

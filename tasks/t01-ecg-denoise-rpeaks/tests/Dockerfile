# Verifier image for t01 (Harbor builds this from the tests/ directory).
# The grader base image provides /opt/grader (root-only venv with bioengbench)
# and an unprivileged `sandbox` user with the same numpy/scipy as the agent.
FROM ghcr.io/mbote-droid/bioengbench-grader-base:0.2.0

COPY . /tests/
RUN chmod 0700 /tests && chmod 0700 /tests/test.sh

# Private evaluation seeds can be mounted or copied over /tests/private_seeds.txt;
# without them the public dev split is used and the score is not held out.

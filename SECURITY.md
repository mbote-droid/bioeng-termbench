# Security policy

BioEng-TermBench runs untrusted code (agent submissions). The grading sandbox
is part of what the benchmark claims, so a way around it is a real defect.

## In scope

* a submission that reads seeds, grader code or hidden truth;
* a submission that writes outside its own output directory or alters a
  trusted record;
* a process that outlives its run, or a limit (memory, CPU, file size,
  process count) that does not hold;
* a submission that earns reward without solving the task, by any means
  other than solving it.

The supported configuration is the one in `docs/THREAT_MODEL.md`: grader as
root, submission as the unprivileged `sandbox` user. Running without the
sandbox user is documented as unsafe, and the command line refuses to do it
without `--trusted`.

## Reporting

Please report privately through GitHub's "Report a vulnerability" button
(Security tab) rather than in a public issue. Include the task, the
submission and the command line. Confirmed issues are fixed, added to the red
team in `validation/redteam/`, and credited unless you prefer otherwise.

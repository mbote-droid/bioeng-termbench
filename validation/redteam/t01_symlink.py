"""Red team: outputs that are symbolic links (to the input, to a system file)."""

import os
import sys

out = sys.argv[sys.argv.index("--outdir") + 1]
inp = sys.argv[sys.argv.index("--input") + 1]
os.makedirs(out, exist_ok=True)
os.symlink(os.path.abspath(inp), os.path.join(out, "ecg_filtered.csv"))
os.symlink("/etc/passwd", os.path.join(out, "rpeaks.csv"))
with open(os.path.join(out, "metrics.json"), "w") as fh:
    fh.write("{}")

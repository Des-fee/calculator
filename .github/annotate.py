"""Turn the interesting tail of a build log into a GitHub annotation (readable via the API)."""
import re
import sys

try:
    lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    lines = ["log file missing"]

# prefer lines around errors, fall back to the tail
idx = [i for i, l in enumerate(lines) if re.search(r"error|Error|FAIL|Traceback|Exception|What went wrong", l)]
keep = []
for i in idx[:12]:
    keep.extend(range(max(0, i - 3), min(len(lines), i + 12)))
keep = sorted(set(keep))
chosen = [lines[i] for i in keep] if keep else []
chosen = (chosen + ["----- tail -----"] + lines[-40:])[-160:]
text = "\n".join(chosen)[-60000:]
text = text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
print("::error title=Build log::" + text)

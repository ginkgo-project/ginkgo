#!/usr/bin/env python3
import subprocess
import sys
import re
import os
import difflib

if len(sys.argv) < 5:
    print(
        "Usage: compare-output.py <working-dir> <path-to-ginkgo-lib> <reference-file> <full-path-to-executable> <args>..."
    )
    sys.exit(1)

cwd = sys.argv[1]
ginkgo_path = sys.argv[2]
reference = open(sys.argv[3]).readlines()
args = sys.argv[4:]
env = os.environ.copy()
if os.name == "nt":
    # PATH is hopefully never empty
    env["PATH"] += ";" + ginkgo_path
else:
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] += ":" + ginkgo_path
    else:
        env["LD_LIBRARY_PATH"] = ginkgo_path

result = subprocess.run(
    args=args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
)

stdout = result.stdout.decode().splitlines()
stderr = result.stderr.decode().splitlines()

if len(stderr) > 0:
    print("FAIL: stderr not empty")
    print("".join(stderr))
    sys.exit(1)

# skip header/footer
stdout = stdout[7:]
reference = reference[5:-4]
# remove everything that looks like a number
remove_pattern = "[-+0-9.]+(e[-+0-9]+)?"
stdout = [re.sub(remove_pattern, "", line.strip()) for line in stdout]
reference = [re.sub(remove_pattern, "", line.strip()) for line in reference]
# compare result
if stdout != reference:
    print("FAIL: stdout differs")
    print("\n".join(difflib.unified_diff(reference, stdout)))
    sys.exit(1)
print("PASS")

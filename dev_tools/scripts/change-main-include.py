#! /usr/bin/env python3
import collections
import sys
import re

filename = sys.argv[1]

suffix = re.compile("(\.cpp|\.hpp|\.cu|\.inc)$")
main_include_re = re.compile("#include\s+<ginkgo/core/([^>]+)>")

Match = collections.namedtuple("Match", ["idx", "line"])

if not suffix.search(filename):
    exit(0)

with open(filename, 'r') as file:
    content = file.readlines()

try:
    first_include = next(Match(idx=i, line=l) for i, l in enumerate(content) if l.startswith("#include"))
except:
    first_include = Match(idx=-1, line="")
if "<ginkgo/core" not in first_include.line:
    exit(0)

try:
    next_idx, next_line = next(Match(idx=i, line=l) for i, l in enumerate(content[first_include.idx + 1:]) if l.strip())
except:
    exit(0)
if next_line.startswith("#if") and next_idx == 0:
    exit(0)
if "<ginkgo/core" in next_line and next_idx == 0:
    exit(0)
if not next_line.startswith("#include") or next_line.startswith('#include "'):
    # Uncertain if the first include is the main include
    print(filename, file=sys.stderr)
    exit(0)

content[first_include.idx] = first_include.line.replace('<', '"').replace('>', '"')
with open(filename, 'w') as file:
        file.writelines(content)

#! /usr/bin/env python3
import collections
import sys
import re

files = sys.argv[1:]

for filename in files:
    suffix = re.compile("(\.cpp|\.hpp|\.cu|\.inc)$")
    main_include_re = re.compile("#include\s+<ginkgo/core/([^>]+)>")

    Match = collections.namedtuple("Match", ["idx", "line"])

    if not suffix.search(filename):
        continue

    with open(filename, 'r') as file:
        content = file.readlines()

    try:
        first_include = next(Match(idx=i, line=l) for i, l in enumerate(content) if l.startswith("#include"))
    except:
        first_include = Match(idx=-1, line="")
    if "<ginkgo/core" not in first_include.line:
        continue

    try:
        next_idx, next_line = next(Match(idx=i, line=l) for i, l in enumerate(content[first_include.idx + 1:]) if l.strip())
    except:
        continue
    if next_line.startswith("#if") and next_idx == 0:
        continue
    if "<ginkgo/core" in next_line and next_idx == 0:
        continue
    if not next_line.startswith("#include") or next_line.startswith('#include "'):
        # Uncertain if the first include is the main include
        print(filename, file=sys.stderr)
        continue

    content[first_include.idx] = first_include.line.replace('<', '"').replace('>', '"')
    with open(filename, 'w') as file:
            file.writelines(content)

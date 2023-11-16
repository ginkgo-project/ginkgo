#! /usr/bin/env python
import re
import sys
from pathlib import Path

files = set(f[len("include/"):] for f in sys.argv[1:] if f.startswith("include/ginkgo/") and f.endswith(".hpp"))
files -= {"ginkgo/ginkgo.hpp", "config.hpp.in", "ginkgo.hpp.in"}

print(files)

if not files:
    exit(0)

ginkgo_main_header_name = Path.cwd() / 'include' / 'ginkgo' / 'ginkgo.hpp'
ginkgo_main_header_name = ginkgo_main_header_name.resolve()

if not ginkgo_main_header_name.exists():
    raise RuntimeError("The main ginkgo.hpp header was not found. This script needs to be run from the root directory.")

header_re = re.compile("#include\s+<([^>]+)>")

with open(ginkgo_main_header_name, 'r') as ginkgo_main_header:
    ginkgo_main_header_content = ginkgo_main_header.read()
ginkgo_includes = set(header_re.findall(ginkgo_main_header_content))

missing_headers = files - ginkgo_includes

if missing_headers:
    ginkgo_main_header_content = ginkgo_main_header_content.splitlines()
    enumerated_includes = [(i, l) for i, l in enumerate(ginkgo_main_header_content) if l.startswith("#include")]
    includes_start, includes_end = min(enumerated_includes, key=lambda t: t[0])[0], max(enumerated_includes, key=lambda t: t[0])[0]
    includes = sorted([t[1] for t in enumerated_includes] + [f"#include <{f}>" for f in missing_headers],
                      key=lambda s: s[s.find("<") + 1:])
    ginkgo_main_header_content = ginkgo_main_header_content[:includes_start] + includes + ginkgo_main_header_content[includes_end + 1:]
    with open(ginkgo_main_header_name, 'w') as ginkgo_main_header:
        ginkgo_main_header.write("\n".join(ginkgo_main_header_content) + "\n")
    exit(1)

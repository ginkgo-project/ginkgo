# SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os, signal
import sys
import time

from cerebras.sdk.client import SdkCompiler

# parse input arguments
if len(sys.argv) != 4:
    exit(1)
try:
    m, n = int(sys.argv[1]), int(sys.argv[2])
except Exception:
    exit(1)
source_path = sys.argv[3]

#pid = os.getpid()  # Get current process ID

# fabric dims
fabdims = "757,996"

with SdkCompiler() as compiler:
    artifact_path = compiler.compile(
        # Path to source files
        source_path,
        # Top level layout file
        "layout.csl",
        # Compiler arguments
        f"--arch=wse2 -o out --fabric-dims={fabdims} --fabric-offsets=4,1 --params=M:{m},grid_size:{n} --memcpy --channels=1 --max-inlined-iterations=200",
        # Output directory
        "."
    )

# write the artifact_path to a json file
with open("artifact_path.json", "w", encoding="utf8") as f:
    json.dump({"artifact_path": artifact_path,}, f)

#os.kill(pid, signal.SIGTERM)  # Sends termination signal to itself

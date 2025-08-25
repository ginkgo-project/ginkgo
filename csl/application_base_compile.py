# SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import sys
import time

from cerebras.sdk.client import SdkCompiler

# parse input arguments
params = None
if len(sys.argv) < 3:
    exit(1)
try:
    if len(sys.argv) >= 4:
        params = sys.argv[3]
except Exception:
    exit(1)
source_path = sys.argv[1]
lib_path = sys.argv[2]

# fabric dims
fabdims = "757,996"

with SdkCompiler() as compiler:
    flags = f"--arch=wse2 -o out --fabric-dims={fabdims} --fabric-offsets=4,1 "
    if params is not None:
        flags += f"--params={params} "
    flags += "--memcpy --channels=1 --max-inlined-iterations=200"
        
    artifact_path = compiler.compile(source_path, "layout.csl", flags, lib_path)
        # Compiler arguments --params=M:{m},grid_size:{n}

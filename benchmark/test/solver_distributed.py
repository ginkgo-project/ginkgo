#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    [
        "-input",
        '[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil", "optimal": {"spmv": "csr-csr"}}]',
    ],
    expected_stdout="distributed_solver.simple.stdout",
    expected_stderr="distributed_solver.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="distributed_solver.simple.stdout",
    expected_stderr="distributed_solver.simple.stderr",
    stdin='[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil", "optimal": {"spmv": "csr-csr"}}]',
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.distributed_solver.json")],
    expected_stdout="distributed_solver.simple.stdout",
    expected_stderr="distributed_solver.simple.stderr",
)

# profiler annotations
test_framework.compare_output(
    [
        "-input",
        '[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil", "optimal": {"spmv": "csr-csr"}}]',
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="distributed_solver.profile.stdout",
    expected_stderr="distributed_solver.profile.stderr",
)

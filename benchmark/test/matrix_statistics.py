#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", '[{"size": 100, "stencil": "7pt"}]'],
    expected_stdout="matrix_statistics.simple.stdout",
    expected_stderr="matrix_statistics.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="matrix_statistics.simple.stdout",
    expected_stderr="matrix_statistics.simple.stderr",
    stdin='[{"size": 100, "stencil": "7pt"}]',
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.mtx.json")],
    expected_stdout="matrix_statistics.simple.stdout",
    expected_stderr="matrix_statistics.simple.stderr",
)

#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", '[{"size": 100, "stencil": "7pt"}]', "-formats", "coo,csr"],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
)

# stdin
test_framework.compare_output(
    ["-formats", "coo,csr"],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
    stdin='[{"size": 100, "stencil": "7pt"}]',
)

# input file
test_framework.compare_output(
    [
        "-input",
        str(test_framework.sourcepath / "input.mtx.json"),
        "-formats",
        "coo,csr",
    ],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
)

# check that all conversions work
test_framework.compare_output(
    [
        "-input",
        '[{"size": 100, "stencil": "7pt"}]',
        "-formats",
        "coo,csr,ell,sellp,hybrid",
    ],
    expected_stdout="conversion.all.stdout",
    expected_stderr="conversion.all.stderr",
)

# profiler annotations
test_framework.compare_output(
    [
        "-input",
        '[{"size": 100, "stencil": "7pt"}]',
        "-formats",
        "coo,csr",
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="conversion.profile.stdout",
    expected_stderr="conversion.profile.stderr",
)

#!/usr/bin/env python3
import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", '[{"size": 100, "stencil": "7pt"}]'],
    expected_stdout="preconditioner.simple.stdout",
    expected_stderr="preconditioner.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="preconditioner.simple.stdout",
    expected_stderr="preconditioner.simple.stderr",
    stdin='[{"size": 100, "stencil": "7pt"}]',
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.mtx.json")],
    expected_stdout="preconditioner.simple.stdout",
    expected_stderr="preconditioner.simple.stderr",
)

# input matrix file
test_framework.compare_output(
    ["-input_matrix", str(test_framework.matrixpath)],
    expected_stdout="preconditioner.matrix.stdout",
    expected_stderr="preconditioner.matrix.stderr",
)

# profiler annotations
test_framework.compare_output(
    [
        "-input",
        '[{"size": 100, "stencil": "7pt"}]',
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="preconditioner.profile.stdout",
    expected_stderr="preconditioner.profile.stderr",
)

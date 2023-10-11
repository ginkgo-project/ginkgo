#!/usr/bin/env python3
import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", '[{"size": 100, "stencil": "7pt"}]'],
    expected_stdout="spmv.simple.stdout",
    expected_stderr="spmv.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="spmv.simple.stdout",
    expected_stderr="spmv.simple.stderr",
    stdin='[{"size": 100, "stencil": "7pt"}]',
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.mtx.json")],
    expected_stdout="spmv.simple.stdout",
    expected_stderr="spmv.simple.stderr",
)

# input matrix file
test_framework.compare_output(
    ["-input_matrix", str(test_framework.matrixpath)],
    expected_stdout="spmv.matrix.stdout",
    expected_stderr="spmv.matrix.stderr",
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
    expected_stdout="spmv.profile.stdout",
    expected_stderr="spmv.profile.stderr",
)

# stdin
test_framework.compare_output(
    ["-reorder", "amd"],
    expected_stdout="spmv.reordered.stdout",
    expected_stderr="spmv.reordered.stderr",
    stdin='[{"size": 100, "stencil": "7pt"}]',
)

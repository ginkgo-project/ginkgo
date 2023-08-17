#!/usr/bin/env python3
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

# input matrix file
test_framework.compare_output(
    ["-input_matrix", str(test_framework.matrixpath)],
    expected_stdout="matrix_statistics.matrix.stdout",
    expected_stderr="matrix_statistics.matrix.stderr",
)

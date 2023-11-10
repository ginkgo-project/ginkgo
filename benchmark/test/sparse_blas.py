#!/usr/bin/env python3
import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-operations", "transpose", "-input",
        '[{"size": 100, "stencil": "7pt"}]'],
    expected_stdout="sparse_blas.simple.stdout",
    expected_stderr="sparse_blas.simple.stderr",
)

# stdin
test_framework.compare_output(
    ["-operations", "transpose"],
    expected_stdout="sparse_blas.simple.stdout",
    expected_stderr="sparse_blas.simple.stderr",
    stdin='[{"size": 100, "stencil": "7pt"}]',
)

# input file
test_framework.compare_output(
    [
        "-operations",
        "transpose",
        "-input",
        str(test_framework.sourcepath / "input.mtx.json"),
    ],
    expected_stdout="sparse_blas.simple.stdout",
    expected_stderr="sparse_blas.simple.stderr",
)

# input matrix file
test_framework.compare_output(
    [
        "-operations",
        "transpose",
        "-input_matrix",
        str(test_framework.matrixpath),
    ],
    expected_stdout="sparse_blas.matrix.stdout",
    expected_stderr="sparse_blas.matrix.stderr",
)

# profiler annotations (transpose has the smallest number of allocations)
test_framework.compare_output(
    [
        "-operations",
        "transpose",
        "-input",
        '[{"size": 100, "stencil": "7pt"}]',
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="sparse_blas.profile.stdout",
    expected_stderr="sparse_blas.profile.stderr",
)

# reordering
test_framework.compare_output(
    ["-operations", "symbolic_cholesky", "-reorder", "amd"],
    expected_stdout="sparse_blas.reordered.stdout",
    expected_stderr="sparse_blas.reordered.stderr",
    stdin='[{"size": 100, "stencil": "7pt"}]',
)

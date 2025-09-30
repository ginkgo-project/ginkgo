#!/usr/bin/env python3
import test_framework


stencil_input = '[{"operator": {"stencil": {"kind": "7pt", "size": 100}}, "operation": "transpose"}]'

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", stencil_input],
    expected_stdout="sparse_blas.simple.stdout",
    expected_stderr="sparse_blas.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="sparse_blas.simple.stdout",
    expected_stderr="sparse_blas.simple.stderr",
    stdin=stencil_input,
)

# input file
test_framework.compare_output(
    [
        "-input",
        str(test_framework.sourcepath / "input.sparse_blas.json"),
    ],
    expected_stdout="sparse_blas.simple.stdout",
    expected_stderr="sparse_blas.simple.stderr",
)

# profiler annotations (transpose has the smallest number of allocations)
test_framework.compare_output(
    [
        "-input",
        stencil_input,
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="sparse_blas.profile.stdout",
    expected_stderr="sparse_blas.profile.stderr",
)

# reordering
test_framework.compare_output(
    [],
    expected_stdout="sparse_blas.reordered.stdout",
    expected_stderr="sparse_blas.reordered.stderr",
    stdin='[{"operator": {"stencil": {"kind": "7pt", "size": 100}}, "operation": "symbolic_cholesky", "reorder": "amd"}]',
)

# complex
test_framework.compare_output(
    ["-input", stencil_input],
    expected_stdout="sparse_blas_dcomplex.simple.stdout",
    expected_stderr="sparse_blas_dcomplex.simple.stderr",
    use_complex=True
)

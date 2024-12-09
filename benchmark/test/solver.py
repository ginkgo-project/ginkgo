#!/usr/bin/env python3
import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", '[{"size": 100, "stencil": "7pt", "optimal": {"spmv": "csr"}}]'],
    expected_stdout="solver.simple.stdout",
    expected_stderr="solver.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="solver.simple.stdout",
    expected_stderr="solver.simple.stderr",
    stdin='[{"size": 100, "stencil": "7pt", "optimal": {"spmv": "csr"}}]',
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.solver.json")],
    expected_stdout="solver.simple.stdout",
    expected_stderr="solver.simple.stderr",
)

# input matrix file
test_framework.compare_output(
    ["-input_matrix", str(test_framework.matrixpath)],
    expected_stdout="solver.matrix.stdout",
    expected_stderr="solver.matrix.stderr",
)

# profiler annotations
test_framework.compare_output(
    [
        "-input",
        '[{"size": 100, "stencil": "7pt", "optimal": {"spmv": "csr"}}]',
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="solver.profile.stdout",
    expected_stderr="solver.profile.stderr",
)

# reordering
test_framework.compare_output(
    ["-reorder", "amd"],
    expected_stdout="solver.reordered.stdout",
    expected_stderr="solver.reordered.stderr",
    stdin='[{"size": 100, "stencil": "7pt", "optimal": {"spmv": "csr"}}]',
)

# complex input
test_framework.compare_output(
    ["-input", '[{"size": 100, "stencil": "7pt", "optimal": {"spmv": "csr"}}]'],
    expected_stdout="solver_dcomplex.simple.stdout",
    expected_stderr="solver_dcomplex.simple.stderr",
    use_complex=True
)

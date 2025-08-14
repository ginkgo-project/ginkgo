#!/usr/bin/env python3
import test_framework
import json

stencil_input = [{"operator": {"stencil": {"name": "7pt", "size": 100}},
                  "preconditioner": {"type": "matrix::Identity"},
                  "format": "csr"}]

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", json.dumps(stencil_input)],
    expected_stdout="preconditioner.simple.stdout",
    expected_stderr="preconditioner.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="preconditioner.simple.stdout",
    expected_stderr="preconditioner.simple.stderr",
    stdin=json.dumps(stencil_input),
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.preconditioner.json")],
    expected_stdout="preconditioner.simple.stdout",
    expected_stderr="preconditioner.simple.stderr",
)

# input matrix file
test_framework.compare_output(
    ["-input_matrix", str(test_framework.matrixpath)],
    expected_stdout="preconditioner.matrix.stdout",
    expected_stderr="preconditioner.matrix.stderr",
)

# set preconditioner works
precond_config = [stencil_input[0] | {
    "preconditioner": {"type": "preconditioner::Jacobi", "max_block_size": 32, "storage_optimization": [0, 0]}}]
test_framework.compare_output(
    [
        "-input",
        json.dumps(precond_config)],
    expected_stdout="preconditioner.precond.stdout",
    expected_stderr="preconditioner.precond.stderr",
)

# profiler annotations
test_framework.compare_output(
    [
        "-input",
        json.dumps(stencil_input),
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="preconditioner.profile.stdout",
    expected_stderr="preconditioner.profile.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="preconditioner.reordered.stdout",
    expected_stderr="preconditioner.reordered.stderr",
    stdin=json.dumps([stencil_input[0] | {"reorder": "amd"}]),
)

# complex
test_framework.compare_output(
    ["-input", json.dumps(stencil_input)],
    expected_stdout="preconditioner_dcomplex.simple.stdout",
    expected_stderr="preconditioner_dcomplex.simple.stderr",
    use_complex=True
)

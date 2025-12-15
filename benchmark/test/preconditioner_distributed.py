#!/usr/bin/env python3
import test_framework

# stdin, a Schwarz preconditioner
test_framework.compare_output_distributed(
    ["-preconditioners", "schwarz-jacobi"],
    expected_stdout="preconditioner_distributed.simple.stdout",
    expected_stderr="preconditioner_distributed.simple.stderr",
    num_procs=3,
    stdin='[{"size": 1000, "stencil": "7pt", "comm_pattern": "stencil"}]',
)

# parameter, a multigrid preconditioner with its levels
test_framework.compare_output_distributed(
    [
        "-preconditioners",
        "mg-pgm",
        "-input",
        '[{"size": 1000, "stencil": "7pt", "comm_pattern": "stencil"}]',
    ],
    expected_stdout="preconditioner_distributed.mg.stdout",
    expected_stderr="preconditioner_distributed.mg.stderr",
    num_procs=3,
)

# input matrix file: every rank reads all of it
test_framework.compare_output_distributed(
    [
        "-preconditioners",
        "schwarz-jacobi",
        "-input_matrix",
        str(test_framework.matrixpath),
    ],
    expected_stdout="preconditioner_distributed.matrix.stdout",
    expected_stderr="preconditioner_distributed.matrix.stderr",
    num_procs=3,
)

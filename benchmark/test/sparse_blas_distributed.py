#!/usr/bin/env python3
import test_framework

# stdin
test_framework.compare_output_distributed(
    ["-operations", "spgemm"],
    expected_stdout="sparse_blas_distributed.simple.stdout",
    expected_stderr="sparse_blas_distributed.simple.stderr",
    num_procs=3,
    stdin='[{"size": 1000, "stencil": "7pt", "comm_pattern": "stencil"}]',
)

# parameter
test_framework.compare_output_distributed(
    [
        "-operations",
        "spgemm",
        "-input",
        '[{"size": 1000, "stencil": "7pt", "comm_pattern": "stencil"}]',
    ],
    expected_stdout="sparse_blas_distributed.simple.stdout",
    expected_stderr="sparse_blas_distributed.simple.stderr",
    num_procs=3,
)

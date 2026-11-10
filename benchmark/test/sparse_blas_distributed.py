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

# the stencil is symmetric, so A * A^T has the pattern of A * A
test_framework.compare_output_distributed(
    ["-operations", "spgemm", "-spgemm_mode", "transposed"],
    expected_stdout="sparse_blas_distributed.simple.stdout",
    expected_stderr="sparse_blas_distributed.simple.stderr",
    num_procs=3,
    stdin='[{"size": 1000, "stencil": "7pt", "comm_pattern": "stencil"}]',
)

# random B
test_framework.compare_output_distributed(
    ["-operations", "spgemm", "-spgemm_mode", "sparse"],
    expected_stdout="sparse_blas_distributed.sparse.stdout",
    expected_stderr="sparse_blas_distributed.simple.stderr",
    num_procs=3,
    stdin='[{"size": 1000, "stencil": "7pt", "comm_pattern": "stencil"}]',
)

# rows of B longer than its columns are full, so C is dense: 1000 * 1000
test_framework.compare_output_distributed(
    [
        "-operations",
        "spgemm",
        "-spgemm_mode",
        "sparse",
        "-spgemm_rowlength",
        "2000",
    ],
    expected_stdout="sparse_blas_distributed.full.stdout",
    expected_stderr="sparse_blas_distributed.simple.stderr",
    num_procs=3,
    stdin='[{"size": 1000, "stencil": "7pt", "comm_pattern": "stencil"}]',
)

test_framework.compare_output_distributed(
    ["-operations", "transpose"],
    expected_stdout="sparse_blas_distributed.transpose.stdout",
    expected_stderr="sparse_blas_distributed.transpose.stderr",
    num_procs=3,
    stdin='[{"size": 1000, "stencil": "7pt", "comm_pattern": "stencil"}]',
)

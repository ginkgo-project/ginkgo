#!/usr/bin/env python3
import test_framework
# check that all input modes work:
# parameter
test_framework.compare_output(["sparse_blas/sparse_blas", "-operations", "transpose", "-input", '[{"size": 100, "stencil": "7pt"}]'],
                              expected_stdout="sparse_blas.simple.stdout",
                              expected_stderr="sparse_blas.simple.stderr")

# stdin
test_framework.compare_output(["sparse_blas/sparse_blas", "-operations", "transpose"],
                              expected_stdout="sparse_blas.simple.stdout",
                              expected_stderr="sparse_blas.simple.stderr",
                              stdin='[{"size": 100, "stencil": "7pt"}]')

# input file
test_framework.compare_output(["sparse_blas/sparse_blas", "-operations", "transpose", "-input", str(test_framework.sourcepath / "input.mtx.json")],
                              expected_stdout="sparse_blas.simple.stdout",
                              expected_stderr="sparse_blas.simple.stderr")

# profiler annotations (transpose has the smallest number of allocations)
test_framework.compare_output(["sparse_blas/sparse_blas", "-operations", "transpose", "-input", '[{"size": 100, "stencil": "7pt"}]', '-profile', '-profiler_hook', 'debug'],
                              expected_stdout="sparse_blas.profile.stdout",
                              expected_stderr="sparse_blas.profile.stderr")

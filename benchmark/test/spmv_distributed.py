#!/usr/bin/env python3
import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output_distributed(
    ["-input", '[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil"}]'],
    expected_stdout="spmv_distributed.simple.stdout",
    expected_stderr="spmv_distributed.simple.stderr",
    num_procs=3,
)

# stdin
test_framework.compare_output_distributed(
    [],
    expected_stdout="spmv_distributed.simple.stdout",
    expected_stderr="spmv_distributed.simple.stderr",
    num_procs=3,
    stdin='[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil"}]',
)

# input file
test_framework.compare_output_distributed(
    ["-input", str(test_framework.sourcepath / "input.distributed_mtx.json")],
    expected_stdout="spmv_distributed.simple.stdout",
    expected_stderr="spmv_distributed.simple.stderr",
    num_procs=3,
)

# profiler annotations
# currently still unstable output and thus disabled
# test_framework.compare_output_distributed(["-input", '[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil"}]', '-profile', '-profiler_hook', 'debug'],
#                                          expected_stdout="spmv_distributed.profile.stdout",
#                                          expected_stderr="spmv_distributed.profile.stderr",
#                                          num_procs=3)

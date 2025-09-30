#!/usr/bin/env python3
import test_framework
import json

case = json.dumps([{"n": 100, "operation": op} for op in ["copy", "axpy", "scal"]])

# check that all input modes work:
# parameter
test_framework.compare_output_distributed(
    ["-input", case],
    expected_stdout="multi_vector_distributed.simple.stdout",
    expected_stderr="multi_vector_distributed.simple.stderr",
    num_procs=3,
)

# stdin
test_framework.compare_output_distributed(
    [],
    expected_stdout="multi_vector_distributed.simple.stdout",
    expected_stderr="multi_vector_distributed.simple.stderr",
    stdin=case,
    num_procs=3,
)

# file
test_framework.compare_output_distributed(
    ["-input", str(test_framework.sourcepath / "input.blas.json")],
    expected_stdout="multi_vector_distributed.simple.stdout",
    expected_stderr="multi_vector_distributed.simple.stderr",
    num_procs=3,
)

# profiler annotations
test_framework.compare_output_distributed(
    ["-input", case, "-profile", "-profiler_hook", "debug"],
    expected_stdout="multi_vector_distributed.profile.stdout",
    expected_stderr="multi_vector_distributed.profile.stderr",
    num_procs=3,
)

# complex
test_framework.compare_output_distributed(
    ["-input", case],
    expected_stdout="multi_vector_distributed_dcomplex.simple.stdout",
    expected_stderr="multi_vector_distributed_dcomplex.simple.stderr",
    num_procs=3,
    use_complex=True
)

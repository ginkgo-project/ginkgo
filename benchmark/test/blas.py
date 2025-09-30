#!/usr/bin/env python3
import test_framework
import json


case = json.dumps([{"n": 100, "operation": op} for op in ["copy", "axpy", "scal"]])

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", case],
    expected_stdout="blas.simple.stdout",
    expected_stderr="blas.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="blas.simple.stdout",
    expected_stderr="blas.simple.stderr",
    stdin=case,
)

# file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.blas.json")],
    expected_stdout="blas.simple.stdout",
    expected_stderr="blas.simple.stderr",
)

# profiler annotations
test_framework.compare_output(
    ["-input", case, "-profile", "-profiler_hook", "debug"],
    expected_stdout="blas.profile.stdout",
    expected_stderr="blas.profile.stderr",
)

# complex
test_framework.compare_output(
    ["-input", case],
    expected_stdout="blas_dcomplex.simple.stdout",
    expected_stderr="blas_dcomplex.simple.stderr",
    use_complex=True
)

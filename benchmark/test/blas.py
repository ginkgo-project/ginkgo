#!/usr/bin/env python3
import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", '[{"n": 100}]'],
    expected_stdout="blas.simple.stdout",
    expected_stderr="blas.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="blas.simple.stdout",
    expected_stderr="blas.simple.stderr",
    stdin='[{"n": 100}]',
)

# file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.blas.json")],
    expected_stdout="blas.simple.stdout",
    expected_stderr="blas.simple.stderr",
)

# profiler annotations
test_framework.compare_output(
    ["-input", '[{"n": 100}]', "-profile", "-profiler_hook", "debug"],
    expected_stdout="blas.profile.stdout",
    expected_stderr="blas.profile.stderr",
)

# adaptive repetitions
test_framework.compare_output(
    [
        "-input", '[{"n": 100}]',
        "-repetitions", "auto",
        "-min_repetitions", "10",
        "-max_repetitions", "20",
    ],
    expected_stdout="blas.auto.stdout",
    expected_stderr="blas.auto.stderr",
)

# ungrouped repetitions
test_framework.compare_output(
    ["-input", '[{"n": 100}]', "-repetition_growth_factor", "1"],
    expected_stdout="blas.ungrouped.stdout",
    expected_stderr="blas.ungrouped.stderr",
)

# complex
test_framework.compare_output(
    ["-input", '[{"n": 100}]'],
    expected_stdout="blas_dcomplex.simple.stdout",
    expected_stderr="blas_dcomplex.simple.stderr",
    use_complex=True
)

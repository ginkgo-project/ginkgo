#!/usr/bin/env python3
import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", '[{"size": 100, "stencil": "7pt"}]', "-formats", "coo,csr"],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
)

# stdin
test_framework.compare_output(
    ["-formats", "coo,csr"],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
    stdin='[{"size": 100, "stencil": "7pt"}]',
)

# input file
test_framework.compare_output(
    [
        "-input",
        str(test_framework.sourcepath / "input.mtx.json"),
        "-formats",
        "coo,csr",
    ],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
)

# input matrixfile
test_framework.compare_output(
    [
        "-input_matrix",
        str(test_framework.matrixpath),
        "-formats",
        "coo,csr",
    ],
    expected_stdout="conversion.matrix.stdout",
    expected_stderr="conversion.matrix.stderr",
)

# check that all conversions work
test_framework.compare_output(
    [
        "-input",
        '[{"size": 100, "stencil": "7pt"}]',
        "-formats",
        "coo,csr,ell,sellp,hybrid",
    ],
    expected_stdout="conversion.all.stdout",
    expected_stderr="conversion.all.stderr",
)

# profiler annotations
test_framework.compare_output(
    [
        "-input",
        '[{"size": 100, "stencil": "7pt"}]',
        "-formats",
        "coo,csr",
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="conversion.profile.stdout",
    expected_stderr="conversion.profile.stderr",
)

# complex
test_framework.compare_output(
    ["-input", '[{"size": 100, "stencil": "7pt"}]', "-formats", "coo,csr"],
    expected_stdout="conversion_dcomplex.simple.stdout",
    expected_stderr="conversion_dcomplex.simple.stderr",
    use_complex=True
)

#!/usr/bin/env python3
import test_framework

stencil_input = '[{"operator": {"stencil": {"kind": "7pt", "size": 100}}, "format": "coo"}]'

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", stencil_input],
    expected_stdout="spmv.simple.stdout",
    expected_stderr="spmv.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="spmv.simple.stdout",
    expected_stderr="spmv.simple.stderr",
    stdin=stencil_input,
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.mtx.json")],
    expected_stdout="spmv.simple.stdout",
    expected_stderr="spmv.simple.stderr",
)

# profiler annotations
test_framework.compare_output(
    [
        "-input",
        stencil_input,
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="spmv.profile.stdout",
    expected_stderr="spmv.profile.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="spmv.reordered.stdout",
    expected_stderr="spmv.reordered.stderr",
    stdin='[{"operator": {"stencil": {"kind": "7pt", "size": 100}}, "format": "csr", "reorder": "amd"}]',
)

# complex
test_framework.compare_output(
    ["-input", stencil_input],
    expected_stdout="spmv_dcomplex.simple.stdout",
    expected_stderr="spmv_dcomplex.simple.stderr",
    use_complex=True
)

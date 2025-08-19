#!/usr/bin/env python3
import test_framework

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", '{"operator": {"stencil": {"kind": "7pt", "size": 100}}}'],
    expected_stdout="matrix_statistics.simple.stdout",
    expected_stderr="matrix_statistics.simple.stderr",
)

# list input
test_framework.compare_output(
    ["-input",
     '[{"operator": {"stencil": {"kind": "7pt", "size": 100}}}, {"operator": {"stencil": {"kind": "27pt", "size": 100}}}]'],
    expected_stdout="matrix_statistics.list.stdout",
    expected_stderr="matrix_statistics.list.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="matrix_statistics.simple.stdout",
    expected_stderr="matrix_statistics.simple.stderr",
    stdin='{"operator": {"stencil": {"kind": "7pt", "size": 100}}}',
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.mtx.json")],
    expected_stdout="matrix_statistics.simple.stdout",
    expected_stderr="matrix_statistics.simple.stderr",
)

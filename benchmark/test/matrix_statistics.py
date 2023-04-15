#!/usr/bin/env python3
import test_framework
# check that all input modes work:
# parameter
test_framework.compare_output(["matrix_statistics/matrix_statistics", "-input", '[{"size": 100, "stencil": "7pt"}]'],
                              expected_stdout="matrix_statistics.simple.stdout",
                              expected_stderr="matrix_statistics.simple.stderr")

# stdin
test_framework.compare_output(["matrix_statistics/matrix_statistics"],
                              expected_stdout="matrix_statistics.simple.stdout",
                              expected_stderr="matrix_statistics.simple.stderr",
                              stdin='[{"size": 100, "stencil": "7pt"}]')

# input file
test_framework.compare_output(["matrix_statistics/matrix_statistics", "-input", '[{"size": 100, "stencil": "7pt"}]'],
                              expected_stdout="matrix_statistics.simple.stdout",
                              expected_stderr="matrix_statistics.simple.stderr")

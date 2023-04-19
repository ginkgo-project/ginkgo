#!/usr/bin/env python3
import test_framework
# check that all input modes work:
# parameter
test_framework.compare_output(["-input", '[{"n": 100}]'],
                              expected_stdout="blas.simple.stdout",
                              expected_stderr="blas.simple.stderr")

# stdin
test_framework.compare_output([],
                              expected_stdout="blas.simple.stdout",
                              expected_stderr="blas.simple.stderr",
                              stdin='[{"n": 100}]')

# file
test_framework.compare_output(["-input", str(test_framework.sourcepath / "input.blas.json")],
                              expected_stdout="blas.simple.stdout",
                              expected_stderr="blas.simple.stderr",
                              stdin='[{"n": 100}]')

# profiler annotations
test_framework.compare_output(["-input", '[{"n": 100}]', '-profile', '-profiler_hook', 'debug'],
                              expected_stdout="blas.profile.stdout",
                              expected_stderr="blas.profile.stderr",
                              stdin='[{"n": 100}]')

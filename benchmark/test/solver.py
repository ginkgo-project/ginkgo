#!/usr/bin/env python3
import test_framework
# check that all input modes work:
# parameter
test_framework.compare_output(["solver/solver", "-input", '[{"size": 100, "stencil": "7pt", "optimal": {"spmv": "csr"}}]'],
                              expected_stdout="solver.simple.stdout",
                              expected_stderr="solver.simple.stderr")

# stdin
test_framework.compare_output(["solver/solver"],
                              expected_stdout="solver.simple.stdout",
                              expected_stderr="solver.simple.stderr",
                              stdin='[{"size": 100, "stencil": "7pt", "optimal": {"spmv": "csr"}}]')

# input file
test_framework.compare_output(["solver/solver", "-input", str(test_framework.sourcepath / "input.solver.json")],
                              expected_stdout="solver.simple.stdout",
                              expected_stderr="solver.simple.stderr")

# profiler annotations
test_framework.compare_output(["solver/solver", "-input", '[{"size": 100, "stencil": "7pt", "optimal": {"spmv": "csr"}}]', '-profile', '-profiler_hook', 'debug'],
                              expected_stdout="solver.profile.stdout",
                              expected_stderr="solver.profile.stderr")

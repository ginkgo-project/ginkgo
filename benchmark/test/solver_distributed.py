#!/usr/bin/env python3
import test_framework
base_flags = ["solver/distributed/solver_distributed"]
# check that all input modes work:
# parameter
test_framework.compare_output(base_flags + ["-input", '[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil", "optimal": {"spmv": "csr-csr"}}]'],
                              expected_stdout="distributed_solver.simple.stdout",
                              expected_stderr="distributed_solver.simple.stderr")

# stdin
test_framework.compare_output(base_flags,
                              expected_stdout="distributed_solver.simple.stdout",
                              expected_stderr="distributed_solver.simple.stderr",
                              stdin='[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil", "optimal": {"spmv": "csr-csr"}}]')

# input file
test_framework.compare_output(base_flags + ["-input", str(test_framework.sourcepath / "input.distributed_solver.json")],
                              expected_stdout="distributed_solver.simple.stdout",
                              expected_stderr="distributed_solver.simple.stderr")

# profiler annotations
test_framework.compare_output(base_flags + ["-input", '[{"size": 100, "stencil": "7pt", "comm_pattern": "stencil", "optimal": {"spmv": "csr-csr"}}]', '-profile', '-profiler_hook', 'debug'],
                              expected_stdout="distributed_solver.profile.stdout",
                              expected_stderr="distributed_solver.profile.stderr")

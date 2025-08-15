#!/usr/bin/env python3
import test_framework
import json

stencil_input = [{"operator": {"stencil": {"kind": "7pt", "local_size": 100}},
                  "solver": {"type": "solver::Cg"},
                  "optimal": {"spmv": {"format": "csr-csr"}}}]

# check that all input modes work:
# parameter
test_framework.compare_output(
    [
        "-input",
        json.dumps(stencil_input),
    ],
    expected_stdout="distributed_solver.simple.stdout",
    expected_stderr="distributed_solver.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="distributed_solver.simple.stdout",
    expected_stderr="distributed_solver.simple.stderr",
    stdin=json.dumps(stencil_input),
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.distributed_solver.json")],
    expected_stdout="distributed_solver.simple.stdout",
    expected_stderr="distributed_solver.simple.stderr",
)

# profiler annotations
test_framework.compare_output(
    [
        "-input",
        json.dumps(stencil_input),
        "-profile",
        "-profiler_hook",
        "debug",
    ],
    expected_stdout="distributed_solver.profile.stdout",
    expected_stderr="distributed_solver.profile.stderr",
)

# complex
test_framework.compare_output(
    [
        "-input",
        json.dumps(stencil_input),
    ],
    expected_stdout="distributed_solver_dcomplex.simple.stdout",
    expected_stderr="distributed_solver_dcomplex.simple.stderr",
    use_complex=True
)

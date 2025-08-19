#!/usr/bin/env python3
import test_framework
import json

stencil_input = {"operator": {"stencil": {"kind": "7pt", "size": 100}},
                 "solver": {"type": "solver::Cg"},
                 "optimal": {"spmv": {"format": "csr"}}}

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", json.dumps(stencil_input)],
    expected_stdout="solver.simple.stdout",
    expected_stderr="solver.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="solver.simple.stdout",
    expected_stderr="solver.simple.stderr",
    stdin=json.dumps(stencil_input),
)

# list input
test_framework.compare_output(
    ["-input", json.dumps(test_framework.config_dict_to_list(
        stencil_input | {"solver": [{"type": "solver::Cg"}, {"type": "solver::Gmres"}]}))],
    expected_stdout="solver.list.stdout",
    expected_stderr="solver.list.stderr",
)

# input file
test_framework.compare_output(
    ["-input", str(test_framework.sourcepath / "input.solver.json")],
    expected_stdout="solver.simple.stdout",
    expected_stderr="solver.simple.stderr",
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
    expected_stdout="solver.profile.stdout",
    expected_stderr="solver.profile.stderr",
)

# reordering
test_framework.compare_output(
    [],
    expected_stdout="solver.reordered.stdout",
    expected_stderr="solver.reordered.stderr",
    stdin=json.dumps(stencil_input | {"optimal": {"spmv": {"format": "csr", "reorder": "amd"}}}),
)

# complex input
test_framework.compare_output(
    ["-input", json.dumps(stencil_input)],
    expected_stdout="solver_dcomplex.simple.stdout",
    expected_stderr="solver_dcomplex.simple.stderr",
    use_complex=True
)

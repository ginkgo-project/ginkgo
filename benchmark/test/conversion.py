#!/usr/bin/env python3
import test_framework
from typing import List, Tuple
import json


def generate_input(operations: List[Tuple[str]]):
    input = [{"operator": {"stencil": {"kind": "7pt", "size": 100}}, "to": to_, "from": from_} for from_, to_ in
             operations]
    return json.dumps(input)


stencil_input = generate_input([("coo", "coo"), ("coo", "csr"), ("csr", "csr"), ("csr", "coo")])
stencil_input_all = generate_input([("coo", "coo"),
                                    ("coo", "csr"),
                                    ("csr", "csr"),
                                    ("csr", "coo"),
                                    ("csr", "ell"),
                                    ("csr", "sellp"),
                                    ("csr", "hybrid"),
                                    ("ell", "ell"),
                                    ("ell", "csr"),
                                    ("sellp", "sellp"),
                                    ("sellp", "csr"),
                                    ("hybrid", "hybrid"),
                                    ("hybrid", "csr")])

# check that all input modes work:
# parameter
test_framework.compare_output(
    ["-input", stencil_input],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
)

# stdin
test_framework.compare_output(
    [],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
    stdin=stencil_input,
)

# input file
test_framework.compare_output(
    [
        "-input",
        str(test_framework.sourcepath / "input.conversion.json")
    ],
    expected_stdout="conversion.simple.stdout",
    expected_stderr="conversion.simple.stderr",
)

# check that all conversions work
test_framework.compare_output(
    [
        "-input",
        stencil_input_all
    ],
    expected_stdout="conversion.all.stdout",
    expected_stderr="conversion.all.stderr",
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
    expected_stdout="conversion.profile.stdout",
    expected_stderr="conversion.profile.stderr",
)

# complex
test_framework.compare_output(
    ["-input", stencil_input],
    expected_stdout="conversion_dcomplex.simple.stdout",
    expected_stderr="conversion_dcomplex.simple.stderr",
    use_complex=True
)

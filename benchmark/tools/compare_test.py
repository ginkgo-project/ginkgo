import json
import compare
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_mismatch(capsys):
    compare.compare_main(
        [
            dir_path + "/../test/reference/blas.simple.stdout",
            dir_path + "/../test/reference/spmv.matrix.stdout",
        ]
    )
    captured = capsys.readouterr()
    ref_out = {"results": [], "outliers": {}}

    ref_err = """WARNING: Key {"n": 100} found in baseline only in context root
WARNING: Key {"filename": ""} found in comparison only in context root
"""
    assert json.loads(captured.out) == ref_out
    assert captured.err == ref_err


def test_simple(capsys):
    compare.compare_main(
        [
            dir_path + "/../test/reference/spmv.matrix.stdout",
            dir_path + "/../test/reference/spmv.matrix.stdout",
        ]
    )
    captured = capsys.readouterr()
    ref_out = {
        "results": [
            {
                "cols": 36,
                "filename": "",
                "nonzeros": 208,
                "rows": 36,
                "spmv": {"coo": {"storage.ratio": 1.0, "time.ratio": 1.0}},
            }
        ],
        "outliers": {},
    }

    assert json.loads(captured.out) == ref_out
    assert captured.err == ""


def test_outliers(capsys):
    compare.compare_main(
        [
            "--outliers",
            dir_path + "/compare_test_input1.json",
            dir_path + "/compare_test_input2.json",
        ]
    )
    captured = capsys.readouterr()
    ref_out = {
        "results": [
            {
                "cols": 36,
                "filename": "mtx",
                "nonzeros": 208,
                "rows": 36,
                "spmv": {
                    "coo": {"storage.ratio": 1.0, "time.ratio": 1.2},
                    "csr": {"storage.ratio": 2.0, "time.ratio": 0.8},
                    "ell": {"storage.ratio": 0.5, "time.ratio": 1.0},
                    "sellp": {"storage.ratio": 1.0, "time.ratio": 1.11},
                    "hybrid": {"storage.ratio": 1.0, "time.ratio": 1.01},
                },
            }
        ],
        "outliers": {
            "spmv/coo/time": [{"value": 1.2, "filename": "mtx"}],
            "spmv/csr/storage": [{"value": 2.0, "filename": "mtx"}],
            "spmv/csr/time": [{"value": 0.8, "filename": "mtx"}],
            "spmv/ell/storage": [{"value": 0.5, "filename": "mtx"}],
            "spmv/sellp/time": [{"value": 1.11, "filename": "mtx"}],
        },
    }

    assert json.loads(captured.out) == ref_out
    assert captured.err == ""


def test_outliers_imited(capsys):
    compare.compare_main(
        [
            "--outliers",
            "--outlier-count",
            "0",
            dir_path + "/compare_test_input1.json",
            dir_path + "/compare_test_input2.json",
        ]
    )
    captured = capsys.readouterr()
    ref_out = {
        "results": [
            {
                "cols": 36,
                "filename": "mtx",
                "nonzeros": 208,
                "rows": 36,
                "spmv": {
                    "coo": {"storage.ratio": 1.0, "time.ratio": 1.2},
                    "csr": {"storage.ratio": 2.0, "time.ratio": 0.8},
                    "ell": {"storage.ratio": 0.5, "time.ratio": 1.0},
                    "sellp": {"storage.ratio": 1.0, "time.ratio": 1.11},
                    "hybrid": {"storage.ratio": 1.0, "time.ratio": 1.01},
                },
            }
        ],
        "outliers": {},
    }

    assert json.loads(captured.out) == ref_out
    assert captured.err == ""


def test_csv(capsys):
    compare.compare_main(
        [
            "--outliers",
            "--output",
            "csv",
            dir_path + "/compare_test_input1.json",
            dir_path + "/compare_test_input2.json",
        ]
    )
    captured = capsys.readouterr()
    ref_out = """benchmark,testcase,ratio
spmv/coo/storage,"{""filename"": ""mtx""}",1.0
spmv/coo/time,"{""filename"": ""mtx""}",1.2
spmv/csr/storage,"{""filename"": ""mtx""}",2.0
spmv/csr/time,"{""filename"": ""mtx""}",0.8
spmv/ell/storage,"{""filename"": ""mtx""}",0.5
spmv/ell/time,"{""filename"": ""mtx""}",1.0
spmv/sellp/storage,"{""filename"": ""mtx""}",1.0
spmv/sellp/time,"{""filename"": ""mtx""}",1.11
spmv/hybrid/storage,"{""filename"": ""mtx""}",1.0
spmv/hybrid/time,"{""filename"": ""mtx""}",1.01


Outliers
benchmark,testcase,ratio
spmv/coo/time,"{""filename"": ""mtx""}",1.2
spmv/csr/storage,"{""filename"": ""mtx""}",2.0
spmv/csr/time,"{""filename"": ""mtx""}",0.8
spmv/ell/storage,"{""filename"": ""mtx""}",0.5
spmv/sellp/time,"{""filename"": ""mtx""}",1.11

"""
    assert captured.out == ref_out
    assert captured.err == ""


def test_md(capsys):
    compare.compare_main(
        [
            "--outliers",
            "--output",
            "markdown",
            dir_path + "/compare_test_input1.json",
            dir_path + "/compare_test_input2.json",
        ]
    )
    captured = capsys.readouterr()
    ref_out = """| benchmark           | testcase            |   ratio |
|:--------------------|:--------------------|--------:|
| spmv/coo/storage    | {"filename": "mtx"} |    1    |
| spmv/coo/time       | {"filename": "mtx"} |    1.2  |
| spmv/csr/storage    | {"filename": "mtx"} |    2    |
| spmv/csr/time       | {"filename": "mtx"} |    0.8  |
| spmv/ell/storage    | {"filename": "mtx"} |    0.5  |
| spmv/ell/time       | {"filename": "mtx"} |    1    |
| spmv/sellp/storage  | {"filename": "mtx"} |    1    |
| spmv/sellp/time     | {"filename": "mtx"} |    1.11 |
| spmv/hybrid/storage | {"filename": "mtx"} |    1    |
| spmv/hybrid/time    | {"filename": "mtx"} |    1.01 |

Outliers
| benchmark        | testcase            |   ratio |
|:-----------------|:--------------------|--------:|
| spmv/coo/time    | {"filename": "mtx"} |    1.2  |
| spmv/csr/storage | {"filename": "mtx"} |    2    |
| spmv/csr/time    | {"filename": "mtx"} |    0.8  |
| spmv/ell/storage | {"filename": "mtx"} |    0.5  |
| spmv/sellp/time  | {"filename": "mtx"} |    1.11 |
"""
    assert captured.out == ref_out
    assert captured.err == ""


def test_complex(capsys):
    compare.compare_main(
        [
            dir_path + "/compare_test_input3.json",
            dir_path + "/compare_test_input3.json",
        ]
    )
    captured = capsys.readouterr()
    ref_out = {
        "results": [
            {
                "filename": "mtx",
                "solver": {
                    "gmres": {
                        "apply": {
                            "components.ratio": {"foo": 1.0},
                            "iterations.ratio": 1.0,
                            "time.ratio": 1.0,
                        },
                        "generate": {"time.ratio": 1.0},
                    }
                },
            },
            {"blas": {"axpy": {"time.ratio": 1.0}}, "k": 2, "m": 3, "n": 1, "r": 4},
            {"size": 100, "spmv": {"csr": {"time.ratio": 1.0}}, "stencil": "7pt"},
        ],
        "outliers": {},
    }

    assert json.loads(captured.out) == ref_out
    assert captured.err == ""

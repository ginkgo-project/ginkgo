#!/usr/bin/env python3
import sys
import json
import argparse
import math
import pandas as pd

parser = argparse.ArgumentParser(description="Compare to Ginkgo benchmark outputs")
parser.add_argument(
    "--outliers", action="store_true", help="List outliers from the results"
)
parser.add_argument(
    "--outlier-threshold",
    type=float,
    default=10,
    help="At what percentage of deviation (above or below) should outliers be reported",
)
parser.add_argument(
    "--outlier-count",
    type=int,
    default=1000,
    help="How many outliers should be reported per benchmark",
)
parser.add_argument("--output", choices=["json", "csv", "markdown"], default="json")
parser.add_argument("baseline")
parser.add_argument("comparison")
args = parser.parse_args()
keys = {"stencil", "size", "filename", "n", "r", "k", "m"}
comparison_keys = {"time", "storage", "iterations"}
suffix = ".ratio"


def key_to_json(key: tuple) -> dict:
    """Restore a JSON output from a key tuple"""
    result = {}
    for key_name, key_val in zip(keys, key):
        if key_val is not None:
            result[key_name] = key_val
    return result


def key_to_str(key: tuple) -> str:
    """Restore a JSON output string from a key tuple"""
    mapped = key_to_json(key)
    if "filename" in mapped.keys():
        return mapped["filename"]
    if "stencil" in mapped.keys():
        return "stencil({}, {})".format(mapped["stencil"], mapped["size"])
    return json.dumps(mapped).replace('"', "")


def parse_json_matrix(filename: str) -> dict:
    """Parse a JSON file into a key -> test_case dict"""
    parsed = json.load(open(filename))
    result = {}
    assert isinstance(parsed, list)
    for case in parsed:
        assert isinstance(case, dict)
        assert not keys.isdisjoint(case.keys())
        dict_key = tuple(case.get(key, None) for key in keys)
        if dict_key in result.keys():
            print(
                "WARNING: Duplicate key {}".format(key_to_str(dict_key)),
                file=sys.stderr,
            )
        result[dict_key] = case
    return result


def warn_on_inconsistent_keys(baseline: dict, comparison: dict, context: str):
    """Print a warning message for non-matching keys between baseline/comparison using the given context string"""
    baseline_only = set(baseline.keys()).difference(comparison.keys())
    comparison_only = set(comparison.keys()).difference(baseline.keys())
    for dict_key in baseline_only:
        print(
            "WARNING: Key {} found in baseline only in context {}".format(
                key_to_str(dict_key), context
            ),
            file=sys.stderr,
        )
    for dict_key in comparison_only:
        print(
            "WARNING: Key {} found in comparison only in context {}".format(
                key_to_str(dict_key), context
            ),
            file=sys.stderr,
        )


def ratio(baseline: int | float, comparison: int | float) -> float:
    """Compares the ratio between baseline and comparison. For runtimes, this is the speedup."""
    return baseline / comparison


def compare_benchmark(baseline: dict, comparison: dict, context: str):
    """Compares a handful of keys and component breakdowns recursively, writing them with a suffix to the output"""
    warn_on_inconsistent_keys(baseline, comparison, context)
    result = {}
    for key in baseline.keys():
        sub_context = "{}.{}".format(context, key)
        if key == "components":
            assert isinstance(baseline[key], dict)
            assert isinstance(comparison[key], dict)
            warn_on_inconsistent_keys(baseline[key], comparison[key], sub_context)
            result[key + suffix] = {
                sub_key: ratio(baseline[key][sub_key], comparison[key][sub_key])
                for sub_key in baseline[key]
            }
        elif isinstance(baseline[key], dict):
            result[key] = compare_benchmark(baseline[key], comparison[key], sub_context)
        elif key in comparison_keys:
            result[key + suffix] = ratio(baseline[key], comparison[key])
    return result


def compare(baseline: dict, comparison: dict, context: str) -> dict:
    """Compares a test case, keeping root-level values and recursing into benchmarks"""
    warn_on_inconsistent_keys(baseline, comparison, context)
    result = {}
    for key in baseline.keys():
        # we don't have lists on the test case root level
        assert not isinstance(baseline[key], list)
        if isinstance(baseline[key], dict):
            benchmark_result = {}
            warn_on_inconsistent_keys(
                baseline[key], comparison[key], "{}.{}".format(context, key)
            )
            for benchmark_name in baseline[key].keys():
                if isinstance(baseline[key][benchmark_name], dict):
                    comparison_result = compare_benchmark(
                        baseline[key][benchmark_name],
                        comparison[key][benchmark_name],
                        "{}.{}.{}".format(context, key, benchmark_name),
                    )
                    if len(comparison_result) > 0:
                        benchmark_result[benchmark_name] = comparison_result
            if len(benchmark_result) > 0:
                result[key] = benchmark_result
        else:
            # everything that's not a dict should only depend on the key in the root level
            if baseline[key] != comparison[key]:
                print(
                    "WARNING: Inconsistent value for {}: {} != {} in context {}".format(
                        key, baseline[key], comparison[key], context
                    ),
                    file=sys.stderr,
                )
            result[key] = baseline[key]
    return result


def extract_benchmark_results(
    input: dict, benchmarks: dict, case_key: tuple, context: str | None
):
    for key, value in input.items():
        benchmark_name = key if context is None else "{}.{}".format(context, key)
        if key in map(lambda x: x + suffix, comparison_keys):
            benchmark_name = benchmark_name[: -len(suffix)]
            if benchmark_name not in benchmarks.keys():
                benchmarks[benchmark_name] = []
            benchmarks[benchmark_name].append((case_key, value))
        elif isinstance(value, dict):
            extract_benchmark_results(value, benchmarks, case_key, benchmark_name)


def is_outlier(value: float):
    return math.fabs(math.log(value)) > math.log(1.0 + args.outlier_threshold / 100)


baseline_json = parse_json_matrix(args.baseline)
comparison_json = parse_json_matrix(args.comparison)
warn_on_inconsistent_keys(baseline_json, comparison_json, "root")

results = {}

for key in baseline_json.keys():
    results[key] = compare(baseline_json[key], comparison_json[key], key_to_str(key))

outliers = {}
if args.outliers:
    benchmarks = {}
    for key, value in results.items():
        extract_benchmark_results(value, benchmarks, key, None)
    for benchmark_name, benchmark_results in benchmarks.items():
        outlier = sorted(
            [
                (case_key, value)
                for case_key, value in benchmark_results
                if is_outlier(value)
            ],
            key=lambda x: math.fabs(math.log(x[1])),
            reverse=True,
        )
        outliers[benchmark_name] = outlier[: min(len(outlier), args.outlier_count)]

if args.output == "json":
    print(
        json.dumps(
            {
                "results": [value for _, value in results.items()],
                "outliers": {
                    key: [
                        {"value": ratio_value} + key_to_json(case_key)
                        for (case_key, ratio_value) in value
                    ]
                    for key, value in outliers.items()
                    if len(value) > 0
                },
            },
            indent=4,
        )
    )
else:
    columns = ["benchmark", "testcase", "ratio"]
    only_first = args.output == "markdown"
    table = pd.DataFrame(
        sum(
            [
                [
                    (
                        key if i == 0 or not only_first else "",
                        key_to_str(value[0]),
                        value[1],
                    )
                    for i, value in enumerate(values)
                ]
                for key, values in benchmarks.items()
            ],
            [],
        ),
        columns=columns,
    )
    if args.output == "csv":
        table.to_csv(sys.stdout, index=False)
    else:
        table.to_markdown(sys.stdout, index=False)
    if args.outliers:
        outlier_table = pd.DataFrame(
            sum(
                [
                    [
                        (
                            key if i == 0 or not only_first else "",
                            key_to_str(value[0]),
                            value[1],
                        )
                        for i, value in enumerate(values)
                    ]
                    for key, values in outliers.items()
                ],
                [],
            ),
            columns=columns,
        )
        if len(outlier_table) > 0:
            print("\n\nOutliers")
            if args.output == "csv":
                outlier_table.to_csv(sys.stdout, index=False)
            else:
                outlier_table.to_markdown(sys.stdout, index=False)
        print()

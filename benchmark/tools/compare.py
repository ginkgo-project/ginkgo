#!/usr/bin/env python3
import sys
import json
import argparse

parser = argparse.ArgumentParser(description="Compare to Ginkgo benchmark outputs")
parser.add_argument("--outlier-threshold")
parser.add_argument("--output")
parser.add_argument("baseline")
parser.add_argument("comparison")
args = parser.parse_args()
keys = {"stencil", "size", "filename", "n", "r", "k", "m"}


def key_to_str(key: tuple) -> str:
    """Restore a JSON output from a key tuple"""
    result = {}
    for key_name, key_val in zip(keys, key):
        if key_val is not None:
            result[key_name] = key_val
    return json.dumps(result)


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
    comparison_keys = {"time", "storage", "iterations"}
    suffix = ".ratio"
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


baseline_json = parse_json_matrix(args.baseline)
comparison_json = parse_json_matrix(args.comparison)
warn_on_inconsistent_keys(baseline_json, comparison_json, "root")

results = []

for key in baseline_json.keys():
    results.append(compare(baseline_json[key], comparison_json[key], key_to_str(key)))

json.dump(results, sys.stdout, indent=4)

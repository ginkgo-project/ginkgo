#!/usr/bin/env bash

# Runs the tests of a Ginkgo build. Must be called from the build directory.
#
# Tests without the `distributed` label run in parallel. The CTest resource
# specification from `test/tools/resource_file_generator` makes sure that
# OpenMP tests get their own CPU cores and device tests their own GPU.
# Tests with the `distributed` label (this includes all MPI tests) run
# afterwards one at a time, since concurrent mpiexec calls can bind their
# ranks to the same cores.
#
# Optional environment variables:
# - CTEST_JOBS: maximum number of tests running at the same time
#   (default: NUM_CORES, or the number of available cores)
# - CTEST_TIMEOUT: timeout per test in seconds (default: 6000)
# - CTEST_JUNIT_DIR: directory for JUnit test reports (requires CTest 3.21+)
# - CTEST_EXTRA_ARGS: additional arguments passed to all ctest calls

set -o pipefail

# nproc would also honor OMP_NUM_THREADS, which is unrelated to the number of
# tests that can run at the same time
jobs="${CTEST_JOBS:-${NUM_CORES:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)}}"
timeout="${CTEST_TIMEOUT:-6000}"

num_tests=$(ctest -N | tail -1 | sed 's/Total Tests: //')
if (( num_tests == 0 )); then
  echo "No tests found"
  exit 1
fi

./test/tools/resource_file_generator > ctest_resources.json || exit 1
cat ctest_resources.json

parallel_junit=()
distributed_junit=()
if [[ -n "${CTEST_JUNIT_DIR}" ]]; then
  ctest_version=$(ctest --version | head -n 1 | awk '{print $3}')
  if printf '3.21\n%s\n' "${ctest_version}" | sort --check=quiet --version-sort; then
    mkdir -p "${CTEST_JUNIT_DIR}"
    parallel_junit=(--output-junit "${CTEST_JUNIT_DIR}/parallel.xml")
    distributed_junit=(--output-junit "${CTEST_JUNIT_DIR}/distributed.xml")
  else
    echo "CTest ${ctest_version} cannot write JUnit reports (requires 3.21)"
  fi
fi

read -r -a extra_args <<< "${CTEST_EXTRA_ARGS}"

status=0
ctest --output-on-failure --timeout "${timeout}" --parallel "${jobs}" \
  --resource-spec-file ctest_resources.json --label-exclude distributed \
  "${parallel_junit[@]}" "${extra_args[@]}" || status=1
ctest --output-on-failure --timeout "${timeout}" --label-regex distributed \
  "${distributed_junit[@]}" "${extra_args[@]}" || status=1
exit ${status}

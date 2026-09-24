#!/usr/bin/env bash

# Configures and builds Ginkgo in ./build and runs its tests. Must be called
# from the source directory, usually on a compute node inside a batch job.
#
# Optional environment variables:
# - MODULES: modules to load before building (e.g. compiler, CUDA, CMake)
# - BUILD_TYPE: CMake build type (default: Release)
# - EXTRA_CMAKE_FLAGS: additional CMake options, e.g. to enable backends
# - BUILD_JOBS: number of parallel build jobs (default: available cores)
# The ctest options are described in .gitlab/run-ctest.sh.

set -e

if [[ -n "${MODULES}" ]]; then
  # batch jobs don't necessarily inherit the module shell function
  if ! command -v module > /dev/null 2>&1 && [[ -n "${LMOD_PKG}" ]]; then
    # shellcheck source=/dev/null
    source "${LMOD_PKG}/init/bash"
  fi
  module purge
  read -r -a modules <<< "${MODULES}"
  module load "${modules[@]}"
  module list
fi

read -r -a cmake_flags <<< "${EXTRA_CMAKE_FLAGS}"
# use Ninja if available, otherwise CMake's default generator
if command -v ninja > /dev/null 2>&1; then
  cmake_flags+=(-G Ninja)
fi
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
  -DGINKGO_DEVEL_TOOLS=OFF -DGINKGO_BUILD_TESTS=ON \
  -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF \
  "${cmake_flags[@]}"
cmake --build build --parallel "${BUILD_JOBS:-$(nproc)}"

cd build
bash ../.gitlab/run-ctest.sh

# CodSpeed benchmarks for Ginkgo, based on CodSpeed's Google Benchmark fork.
#
# This is a standalone project that builds a minimal Ginkgo (reference
# executor only) and a set of Google Benchmark executables on top of it.
#
#   cmake -S benchmark/codspeed -B build-codspeed \
#         -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCODSPEED_MODE=simulation
#   cmake --build build-codspeed -j
#   ./build-codspeed/ginkgo_codspeed_bench
cmake_minimum_required(VERSION 3.16)

project(GinkgoCodSpeedBenchmarks LANGUAGES CXX)

include(FetchContent)

if(NOT CMAKE_BUILD_TYPE)
    # optimized code with debug symbols, needed for useful profiles
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# Build only what the benchmarks need: the core library and the
# single-threaded reference kernels in double precision.
set(GINKGO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GINKGO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GINKGO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(GINKGO_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(GINKGO_DEVEL_TOOLS OFF CACHE BOOL "" FORCE)
set(GINKGO_BUILD_REFERENCE ON CACHE BOOL "" FORCE)
set(GINKGO_BUILD_OMP OFF CACHE BOOL "" FORCE)
set(GINKGO_BUILD_MPI OFF CACHE BOOL "" FORCE)
set(GINKGO_BUILD_CUDA OFF CACHE BOOL "" FORCE)
set(GINKGO_BUILD_HIP OFF CACHE BOOL "" FORCE)
set(GINKGO_BUILD_SYCL OFF CACHE BOOL "" FORCE)
set(GINKGO_MIXED_PRECISION OFF CACHE BOOL "" FORCE)
set(GINKGO_ENABLE_HALF OFF CACHE BOOL "" FORCE)
set(GINKGO_ENABLE_BFLOAT16 OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ginkgo)

# Benchmark names are reported relative to the repository root
get_filename_component(
    _ginkgo_root
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ABSOLUTE
)
set(CODSPEED_ROOT_DIR
    "${_ginkgo_root}"
    CACHE PATH
    "Root directory for CodSpeed"
)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/CodSpeedHQ/codspeed-cpp
    GIT_TAG v2.4.0
    SOURCE_SUBDIR
    google_benchmark
)
# The CodSpeed instrumentation hooks need to be linked statically into the
# benchmark executable, independently of how Ginkgo itself is built.
set(_ginkgo_codspeed_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(google_benchmark)
set(BUILD_SHARED_LIBS ${_ginkgo_codspeed_shared_libs})

add_executable(
    ginkgo_codspeed_bench
    main.cpp
    spmv.cpp
    blas.cpp
    preconditioner.cpp
    solver.cpp
)
target_compile_features(ginkgo_codspeed_bench PRIVATE cxx_std_17)
target_link_libraries(
    ginkgo_codspeed_bench
    PRIVATE Ginkgo::ginkgo benchmark::benchmark
)

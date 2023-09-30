ginkgo_add_typed_benchmark_executables(spmv "YES" spmv.cpp)
# TODO: move to all benchmark
if (GINKGO_ENABLE_HALF)
    ginkgo_add_single_benchmark_executable(
        "spmv_half" "YES" "GKO_BENCHMARK_USE_HALF_PRECISION" "h" spmv.cpp)
endif()
if(GINKGO_BUILD_MPI)
    add_subdirectory(distributed)
endif()

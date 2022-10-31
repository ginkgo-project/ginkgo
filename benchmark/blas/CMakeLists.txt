ginkgo_add_typed_benchmark_executables(blas "NO" blas.cpp)
if(GINKGO_BUILD_MPI)
    add_subdirectory(distributed)
endif()

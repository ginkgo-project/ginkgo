ginkgo_add_typed_benchmark_executables(preconditioner "NO" preconditioner.cpp)
if(GINKGO_BUILD_MPI)
    add_subdirectory(distributed)
endif()

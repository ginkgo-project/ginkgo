ginkgo_add_typed_benchmark_executables(spmv "YES" spmv.cpp)
if(GINKGO_BUILD_MPI)
    add_subdirectory(distributed)
endif()

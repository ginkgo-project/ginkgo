ginkgo_add_typed_benchmark_executables(solver "YES" solver.cpp)
if(GINKGO_BUILD_MPI)
    add_subdirectory(distributed)
endif()

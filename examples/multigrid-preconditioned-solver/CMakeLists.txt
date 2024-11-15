cmake_minimum_required(VERSION 3.16)
project(multigrid-preconditioned-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif()

add_executable(multigrid-preconditioned-solver multigrid-preconditioned-solver.cpp)
target_link_libraries(multigrid-preconditioned-solver Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/mat0.bin data/mat0.bin COPYONLY)
configure_file(data/rhs0.bin data/rhs0.bin COPYONLY)
configure_file(data/x0.bin data/x0.bin COPYONLY)

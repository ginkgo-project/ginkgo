cmake_minimum_required(VERSION 3.16)
project(ir-ilu-preconditioned-solver)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 2.0.0 REQUIRED)
endif()

add_executable(ir-ilu-preconditioned-solver ir-ilu-preconditioned-solver.cpp)
target_link_libraries(ir-ilu-preconditioned-solver Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)

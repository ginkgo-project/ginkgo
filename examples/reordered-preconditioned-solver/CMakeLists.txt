cmake_minimum_required(VERSION 3.16)
project(reordered-preconditioned-solver)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 2.0.0 REQUIRED)
endif()
add_executable(
    reordered-preconditioned-solver
    reordered-preconditioned-solver.cpp
)
target_link_libraries(reordered-preconditioned-solver Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)

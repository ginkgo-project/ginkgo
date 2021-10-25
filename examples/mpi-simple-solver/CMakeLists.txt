cmake_minimum_required(VERSION 3.9)
project(mpi-simple-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
    find_package(MPI REQUIRED)
endif()

add_executable(mpi-simple-solver mpi-simple-solver.cpp)
target_link_libraries(mpi-simple-solver Ginkgo::ginkgo)
target_link_libraries(mpi-simple-solver MPI::MPI_CXX)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)

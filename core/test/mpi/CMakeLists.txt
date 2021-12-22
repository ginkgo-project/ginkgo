add_library(gtest_mpi_main "")
target_sources(gtest_mpi_main
    PRIVATE
    gtest/mpi_listener.cpp)
find_package(MPI REQUIRED)
target_link_libraries(gtest_mpi_main PRIVATE GTest::GTest MPI::MPI_CXX)
add_subdirectory(base)
add_subdirectory(distributed)

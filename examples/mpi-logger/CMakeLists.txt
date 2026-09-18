cmake_minimum_required(VERSION 3.16)
project(mpi-logger)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_executable(mpi-logger mpi-logger.cpp)
target_link_libraries(mpi-logger Ginkgo::ginkgo)

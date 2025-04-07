cmake_minimum_required(VERSION 3.16)
project(distributed-spmv)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.10.0 REQUIRED)
endif()

add_executable(distributed-spmv distributed-spmv.cpp)
target_link_libraries(distributed-spmv Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)

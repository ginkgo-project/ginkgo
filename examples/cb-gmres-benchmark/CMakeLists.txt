cmake_minimum_required(VERSION 3.9)
project(cb-gmres-benchmark)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

add_executable(cb-gmres-benchmark cb-gmres-benchmark.cpp)
target_link_libraries(cb-gmres-benchmark Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file("${Ginkgo_SOURCE_DIR}/matrices/test/ani1.mtx" data/A.mtx COPYONLY)

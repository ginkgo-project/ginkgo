cmake_minimum_required(VERSION 3.16)
project(cb-gmres)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()

add_executable(cb-gmres cb-gmres.cpp)
target_link_libraries(cb-gmres Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file("${Ginkgo_SOURCE_DIR}/matrices/test/ani1.mtx" data/A.mtx COPYONLY)

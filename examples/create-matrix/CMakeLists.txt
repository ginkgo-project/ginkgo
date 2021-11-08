cmake_minimum_required(VERSION 3.9)
project(create-matrix)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

add_executable(create-matrix create-matrix.cpp)
target_link_libraries(create-matrix Ginkgo::ginkgo)

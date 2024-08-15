cmake_minimum_required(VERSION 3.16)
project(batched-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif()

add_executable(batched-matrix-free batched-matrix-free.cpp)
target_link_libraries(batched-matrix-free Ginkgo::ginkgo)

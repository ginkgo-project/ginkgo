cmake_minimum_required(VERSION 3.9)
project(batched-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

add_executable(batched-solver batched-solver.cpp)
target_link_libraries(batched-solver Ginkgo::ginkgo)

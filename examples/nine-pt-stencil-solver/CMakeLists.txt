cmake_minimum_required(VERSION 3.16)
project(nine-pt-stencil-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()

add_executable(nine-pt-stencil-solver nine-pt-stencil-solver.cpp)
target_link_libraries(nine-pt-stencil-solver Ginkgo::ginkgo)

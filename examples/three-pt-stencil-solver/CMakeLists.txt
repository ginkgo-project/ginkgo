cmake_minimum_required(VERSION 3.9)
project(three-pt-stencil-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.6.0 REQUIRED)
endif()

add_executable(three-pt-stencil-solver three-pt-stencil-solver.cpp)
target_link_libraries(three-pt-stencil-solver Ginkgo::ginkgo)

cmake_minimum_required(VERSION 3.16)
project(distributed-multigrid-preconditioned-solver-customized)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.10.0 REQUIRED)
endif()

add_executable(distributed-multigrid-preconditioned-solver-customized distributed-multigrid-preconditioned-solver-customized.cpp)
target_link_libraries(distributed-multigrid-preconditioned-solver-customized Ginkgo::ginkgo)

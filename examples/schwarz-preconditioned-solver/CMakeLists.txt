cmake_minimum_required(VERSION 3.9)
project(schwarz-preconditioned-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

add_executable(schwarz-preconditioned-solver schwarz-preconditioned-solver.cpp)
target_link_libraries(schwarz-preconditioned-solver Ginkgo::ginkgo)

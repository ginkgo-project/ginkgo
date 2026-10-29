cmake_minimum_required(VERSION 3.16)
project(distributed-multigrid-reuse)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_executable(distributed-multigrid-reuse distributed-multigrid-reuse.cpp)
target_link_libraries(distributed-multigrid-reuse Ginkgo::ginkgo)

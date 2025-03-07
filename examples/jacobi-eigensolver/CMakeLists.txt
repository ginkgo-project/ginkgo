cmake_minimum_required(VERSION 3.16)
project(jacobi-eigensolver)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.10.0 REQUIRED)
endif()

add_executable(jacobi-eigensolver jacobi-eigensolver.cpp)
target_link_libraries(jacobi-eigensolver Ginkgo::ginkgo)

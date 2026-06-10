cmake_minimum_required(VERSION 3.16)
project(scale-correction-mg-solver)

if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_executable(scale-correction-mg-solver scale-correction-mg-solver.cpp)
target_link_libraries(scale-correction-mg-solver Ginkgo::ginkgo)

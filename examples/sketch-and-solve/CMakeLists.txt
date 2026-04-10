cmake_minimum_required(VERSION 3.16)
project(sketch-and-solve)

if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.1
)
FetchContent_MakeAvailable(cxxopts)

add_executable(sketch-and-solve sketch-and-solve.cpp)
target_link_libraries(sketch-and-solve Ginkgo::ginkgo cxxopts::cxxopts)

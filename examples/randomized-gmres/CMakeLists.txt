cmake_minimum_required(VERSION 3.16)
project(randomized-gmres)

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

add_executable(randomized-gmres randomized-gmres.cpp)
target_link_libraries(randomized-gmres Ginkgo::ginkgo cxxopts::cxxopts)

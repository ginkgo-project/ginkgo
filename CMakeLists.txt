cmake_minimum_required(VERSION 3.1)
project(MSparse)

option(BUILD_TESTS "Generate build files for unit tests" ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

if (BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(third_party)  # Third-party tools and libraries
add_subdirectory(core)         # Core MSparse classes


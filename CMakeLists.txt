cmake_minimum_required(VERSION 3.1)
project(Ginkgo)

option(BUILD_TESTS "Generate build files for unit tests" ON)
option(DEVEL_TOOLS "Add development tools to the build system" ON)
option(BUILD_REFERENCE "Compile reference CPU kernels" OFF)
option(BUILD_CPU "Compile kernels for CPUs" OFF)
option(BUILD_GPU "Compile kernels for NVIDIA GPUs" OFF)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(third_party)  # Third-party tools and libraries
add_subdirectory(core)         # Core Ginkgo types and top-level functions


if(DEVEL_TOOLS)
    set(LICENSE_SCRIPT ${CMAKE_SOURCE_DIR}/dev_tools/scripts/add_license.sh)

    add_custom_target(add_license
        COMMAND ${LICENSE_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_dependencies(format add_license)
endif()

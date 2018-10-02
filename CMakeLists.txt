if(WIN32)
    cmake_minimum_required(VERSION 3.9)
else()
    cmake_minimum_required(VERSION 3.8)
endif()

project(Ginkgo LANGUAGES CXX VERSION 0.0.0)
set(Ginkgo_VERSION_TAG "develop")
set(PROJECT_VERSION_TAG ${Ginkgo_VERSION_TAG})

option(DEVEL_TOOLS "Add development tools to the build system" ON)
option(BUILD_TESTS "Generate build files for unit tests" ON)
option(BUILD_EXAMPLES "Build Ginkgo's examples" ON)
option(BUILD_BENCHMARKS "Build Ginkgo's benchmarks" ON)
option(BUILD_REFERENCE "Compile reference CPU kernels" OFF)
option(BUILD_OMP "Compile OpenMP kernels for CPU" OFF)
option(BUILD_CUDA "Compile kernels for NVIDIA GPUs" OFF)
option(BUILD_DOC "Generate documentation" OFF)
option(BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)
option(SET_CUDA_HOST_COMPILER "Propagate selected C++ compiler to CUDA" OFF)
option(SKIP_DEPENDENCY_UPDATE
       "Do not update dependencies each time the project is rebuilt" ON)
set(CUDA_ARCHITECTURES "Auto" CACHE STRING
    "A list of target NVIDIA GPU achitectures. See README.md for more detail.")

include(cmake/create_test.cmake)
include(cmake/install_helpers.cmake)

if(BUILD_TESTS AND (BUILD_CUDA OR BUILD_OMP))
    message(STATUS "BUILD_TESTS is ON, enabling BUILD_REFERENCE")
    set(BUILD_REFERENCE ON CACHE BOOL "Compile reference CPU kernels" FORCE)
endif()

if(BUILD_CUDA)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(BUILD_TESTS)
    enable_testing()
endif()


# Find important header files, store the definitions in include/config.h.in
# For details, see https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/How-To-Write-Platform-Checks
include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/config.hpp)
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include)


add_subdirectory(third_party)    # Third-party tools and libraries
add_subdirectory(core)           # Core Ginkgo types and top-level functions
if (BUILD_REFERENCE)
    add_subdirectory(reference)  # Reference kernel implementations
endif()
if (BUILD_OMP)
    add_subdirectory(omp)        # High-performance omp kernels
endif()
if(BUILD_CUDA)
    add_subdirectory(cuda)       # High-performance kernels for NVIDIA GPUs
endif()

ginkgo_install()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(DEVEL_TOOLS)
    set(LICENSE_SCRIPT ${CMAKE_SOURCE_DIR}/dev_tools/scripts/add_license.sh)

    add_custom_target(add_license
        COMMAND ${LICENSE_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_dependencies(format add_license)
endif()

if(BUILD_DOC)
	add_subdirectory(doc)
endif()

if(WIN32)
    cmake_minimum_required(VERSION 3.9)
else()
    cmake_minimum_required(VERSION 3.8)
endif()

project(Ginkgo LANGUAGES C CXX VERSION 0.0.0)
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
option(SKIP_DEPENDENCY_UPDATE
       "Do not update dependencies each time the project is rebuilt" ON)
set(GINKGO_VERBOSE_LEVEL "1" CACHE STRING
       "Verbosity level. Put 0 to turn off. 1 activates a few important messages.")
set(CUDA_ARCHITECTURES "Auto" CACHE STRING
    "A list of target NVIDIA GPU achitectures. See README.md for more detail.")

include(cmake/create_test.cmake)
include(cmake/install_helpers.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(BUILD_TESTS AND (BUILD_CUDA OR BUILD_OMP))
    message(STATUS "BUILD_TESTS is ON, enabling BUILD_REFERENCE")
    set(BUILD_REFERENCE ON CACHE BOOL "Compile reference CPU kernels" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(BUILD_TESTS)
    set(MEMORYCHECK_SUPPRESSIONS_FILE
        "${PROJECT_SOURCE_DIR}/dev_tools/valgrind/suppressions.supp"
        CACHE FILEPATH "" FORCE)
    include(CTest)
    enable_testing()
endif()


# Find important header files, store the definitions in include/ginkgo/config.h.in
# For details, see https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/How-To-Write-Platform-Checks
include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)


# Automatically find PAPI and search for the required 'sde' component
find_package(PAPI OPTIONAL_COMPONENTS sde)
if (PAPI_sde_FOUND)
    set(GKO_HAVE_PAPI_SDE 1)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/ginkgo/config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/ginkgo/config.hpp)

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
  DESTINATION "include"
  FILES_MATCHING PATTERN "*.hpp"
  )
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/"
  DESTINATION "include"
  FILES_MATCHING PATTERN "*.hpp"
  )


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

    # Generate the global `ginkgo/ginkgo.hpp` header with every call of make
    add_custom_target(generate_ginkgo_header ALL
        COMMAND ${CMAKE_SOURCE_DIR}/dev_tools/scripts/update_ginkgo_header.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

if(BUILD_DOC)
	add_subdirectory(doc)
endif()

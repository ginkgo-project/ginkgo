cmake_minimum_required(VERSION 3.9)

project(Ginkgo LANGUAGES C CXX VERSION 0.0.0)
set(Ginkgo_VERSION_TAG "develop")
set(PROJECT_VERSION_TAG ${Ginkgo_VERSION_TAG})

# Load CMake helpers and modules
include(cmake/build_helpers.cmake)
include(cmake/create_test.cmake)
include(cmake/install_helpers.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")


# Ginkgo configuration options
option(GINKGO_DEVEL_TOOLS "Add development tools to the build system" ON)
option(GINKGO_BUILD_TESTS "Generate build files for unit tests" ON)
option(GINKGO_BUILD_EXAMPLES "Build Ginkgo's examples" ON)
option(GINKGO_BUILD_BENCHMARKS "Build Ginkgo's benchmarks" ON)
option(GINKGO_BUILD_REFERENCE "Compile reference CPU kernels" OFF)
option(GINKGO_BUILD_OMP "Compile OpenMP kernels for CPU" OFF)
option(GINKGO_BUILD_CUDA "Compile kernels for NVIDIA GPUs" OFF)
option(GINKGO_BUILD_DOC "Generate documentation" OFF)
option(GINKGO_SKIP_DEPENDENCY_UPDATE
       "Do not update dependencies each time the project is rebuilt" ON)
option(GINKGO_EXPORT_BUILD_DIR
    "Make Ginkgo export its build directory to the CMake package registry."
    OFF)
set(GINKGO_VERBOSE_LEVEL "1" CACHE STRING
    "Verbosity level. Put 0 to turn off. 1 activates a few important messages.")
set(GINKGO_CUDA_ARCHITECTURES "Auto" CACHE STRING
    "A list of target NVIDIA GPU achitectures. See README.md for more detail.")
option(BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)

if(GINKGO_BUILD_TESTS AND (GINKGO_BUILD_CUDA OR GINKGO_BUILD_OMP))
    message(STATUS "GINKGO_BUILD_TESTS is ON, enabling GINKGO_BUILD_REFERENCE")
    set(GINKGO_BUILD_REFERENCE ON CACHE BOOL "Compile reference CPU kernels" FORCE)
endif()


if(GINKGO_BUILD_TESTS)
    set(MEMORYCHECK_SUPPRESSIONS_FILE
        "${Ginkgo_SOURCE_DIR}/dev_tools/valgrind/suppressions.supp"
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

configure_file(${Ginkgo_SOURCE_DIR}/include/ginkgo/config.hpp.in
  ${Ginkgo_BINARY_DIR}/include/ginkgo/config.hpp)


add_subdirectory(third_party)    # Third-party tools and libraries
add_subdirectory(core)           # Core Ginkgo types and top-level functions
if (GINKGO_BUILD_REFERENCE)
    add_subdirectory(reference)  # Reference kernel implementations
endif()
if (GINKGO_BUILD_OMP)
    add_subdirectory(omp)        # High-performance omp kernels
endif()
if(GINKGO_BUILD_CUDA)
    add_subdirectory(cuda)       # High-performance kernels for NVIDIA GPUs
endif()

ginkgo_install()

if(GINKGO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(GINKGO_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(GINKGO_DEVEL_TOOLS)
    add_custom_target(add_license
        COMMAND ${Ginkgo_SOURCE_DIR}/dev_tools/scripts/add_license.sh
        WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR})
    add_dependencies(format add_license)
endif()

# Generate the global `ginkgo/ginkgo.hpp` header with every call of make
add_custom_target(generate_ginkgo_header ALL
        COMMAND ${Ginkgo_SOURCE_DIR}/dev_tools/scripts/update_ginkgo_header.sh
        WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR})

if(GINKGO_BUILD_DOC)
    add_subdirectory(doc)
endif()

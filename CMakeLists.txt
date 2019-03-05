cmake_minimum_required(VERSION 3.9)

project(Ginkgo LANGUAGES C CXX VERSION 0.0.0 DESCRIPTION "Numerical linear algebra software")
set(Ginkgo_VERSION_TAG "develop")
set(PROJECT_VERSION_TAG ${Ginkgo_VERSION_TAG})

# Load CMake helpers and modules
include(cmake/build_helpers.cmake)
include(cmake/create_test.cmake)
include(cmake/install_helpers.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

include(cmake/build_type_helpers.cmake)

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
set(GINKGO_COMPILER_FLAGS "-Wpedantic" CACHE STRING
  "Set the required CXX compiler flags, mainly used for warnings. Current default is `-Wpedantic`")
set(GINKGO_CUDA_COMPILER_FLAGS "" CACHE STRING
  "Set the required NVCC compiler flags, mainly used for warnings. Current default is an empty string")
set(GINKGO_CUDA_ARCHITECTURES "Auto" CACHE STRING
    "A list of target NVIDIA GPU achitectures. See README.md for more detail.")
option(BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)

if(GINKGO_BUILD_TESTS AND (GINKGO_BUILD_CUDA OR GINKGO_BUILD_OMP))
    message(STATUS "GINKGO_BUILD_TESTS is ON, enabling GINKGO_BUILD_REFERENCE")
    set(GINKGO_BUILD_REFERENCE ON CACHE BOOL "Compile reference CPU kernels" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()


if(GINKGO_BUILD_TESTS)
    # Configure CTest
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestCustom.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake @ONLY)

    enable_testing()
    include(CTest)
endif()


# Find important header files, store the definitions in include/ginkgo/config.h.in
# For details, see https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/How-To-Write-Platform-Checks
include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)

# Automatically find PAPI and search for the required 'sde' component
set(GINKGO_HAVE_PAPI_SDE 0)
find_package(PAPI OPTIONAL_COMPONENTS sde)
if (PAPI_sde_FOUND)
    set(GINKGO_HAVE_PAPI_SDE 1)
endif()

configure_file(${Ginkgo_SOURCE_DIR}/include/ginkgo/config.hpp.in
    ${Ginkgo_BINARY_DIR}/include/ginkgo/config.hpp @ONLY)

# Try to find the third party packages before using our subdirectories
include(cmake/package_helpers.cmake)
ginkgo_find_package(GTest)
ginkgo_find_package(gflags)
ginkgo_find_package(RapidJSON)
add_subdirectory(third_party)    # Third-party tools and libraries

# Ginkgo core libraries
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

# Non core directories and targets
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

# Installation
include(cmake/information_helpers.cmake)
ginkgo_interface_information()
ginkgo_git_information()

configure_file(${Ginkgo_SOURCE_DIR}/cmake/ginkgo.pc.in
    ${Ginkgo_BINARY_DIR}/ginkgo.pc @ONLY)

ginkgo_install()

add_custom_target(test_install
    COMMAND ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} -H${Ginkgo_SOURCE_DIR}/test_install
        -B${Ginkgo_BINARY_DIR}/test_install
        -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_CONFIG_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${Ginkgo_BINARY_DIR}/test_install
    COMMAND ${Ginkgo_BINARY_DIR}/test_install/test_install
    COMMENT "Running a test on the installed binaries. This requires running `(sudo) make install` first.")

# Setup CPack and ensure we have a debug postfix
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX "d")
endif()

set(CPACK_PACKAGE_DESCRIPTION_FILE "${Ginkgo_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${Ginkgo_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_ICON "${Ginkgo_SOURCE_DIR}/assets/logo.png")
set(CPACK_PACKAGE_CONTACT "ginkgo.library@gmail.com")
include(CPack)

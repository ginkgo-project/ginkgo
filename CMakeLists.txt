cmake_minimum_required(VERSION 3.9)

project(Ginkgo LANGUAGES C CXX VERSION 1.1.1 DESCRIPTION "A numerical linear algebra library targeting many-core architectures")
set(Ginkgo_VERSION_TAG "develop")
set(PROJECT_VERSION_TAG ${Ginkgo_VERSION_TAG})
include(CheckLanguage)

# Determine which languages can be compiled
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(GINKGO_HAS_CUDA ON)
else()
    set(GINKGO_HAS_CUDA OFF)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(GINKGO_HAS_OMP ON)
else()
    set(GINKGO_HAS_OMP OFF)
endif()

# TODO: auto-detect HIP
set(GINKGO_HAS_HIP OFF)

# Ginkgo configuration options
option(GINKGO_DEVEL_TOOLS "Add development tools to the build system" ON)
option(GINKGO_BUILD_TESTS "Generate build files for unit tests" ON)
option(GINKGO_BUILD_EXAMPLES "Build Ginkgo's examples" ON)
option(GINKGO_BUILD_BENCHMARKS "Build Ginkgo's benchmarks" ON)
option(GINKGO_BUILD_REFERENCE "Compile reference CPU kernels" ON)
option(GINKGO_BUILD_OMP "Compile OpenMP kernels for CPU" ${GINKGO_HAS_OMP})
option(GINKGO_BUILD_CUDA "Compile kernels for NVIDIA GPUs" ${GINKGO_HAS_CUDA})
option(GINKGO_BUILD_HIP "Compile kernels for AMD or NVIDIA GPUs" ${GINKGO_HAS_HIP})
option(GINKGO_BUILD_DOC "Generate documentation" OFF)
option(GINKGO_SKIP_DEPENDENCY_UPDATE
    "Do not update dependencies each time the project is rebuilt" ON)
option(GINKGO_EXPORT_BUILD_DIR
    "Make Ginkgo export its build directory to the CMake package registry."
    OFF)
option(GINKGO_WITH_CLANG_TIDY "Make Ginkgo call `clang-tidy` to find programming issues." OFF)
option(GINKGO_WITH_IWYU "Make Ginkgo call `iwyu` (Include What You Use) to find include issues." OFF)
set(GINKGO_VERBOSE_LEVEL "1" CACHE STRING
    "Verbosity level. Put 0 to turn off. 1 activates a few important messages.")
if(MSVC)
    set(GINKGO_COMPILER_FLAGS "" CACHE STRING
        "Set the required CXX compiler flags, mainly used for warnings. Current default is ``")
else()
    set(GINKGO_COMPILER_FLAGS "-Wpedantic" CACHE STRING
        "Set the required CXX compiler flags, mainly used for warnings. Current default is `-Wpedantic`")
endif()
set(GINKGO_CUDA_COMPILER_FLAGS "" CACHE STRING
    "Set the required NVCC compiler flags, mainly used for warnings. Current default is an empty string")
set(GINKGO_CUDA_ARCHITECTURES "Auto" CACHE STRING
    "A list of target NVIDIA GPU achitectures. See README.md for more detail.")
option(GINKGO_CUDA_DEFAULT_HOST_COMPILER "Tell Ginkgo to not automatically set the CUDA host compiler" OFF)
set(GINKGO_HIP_COMPILER_FLAGS "" CACHE STRING
    "Set the required HIP compiler flags. Current default is an empty string.")
set(GINKGO_HIP_NVCC_COMPILER_FLAGS "" CACHE STRING
    "Set the required HIP nvcc compiler flags. Current default is an empty string.")
set(GINKGO_HIP_HCC_COMPILER_FLAGS "" CACHE STRING
    "Set the required HIPC HCC compiler flags. Current default is an empty string.")
set(GINKGO_HIP_AMDGPU "" CACHE STRING
    "The amdgpu_target(s) variable passed to hipcc. The default is none (auto).")
option(GINKGO_JACOBI_FULL_OPTIMIZATIONS "Use all the optimizations for the CUDA Jacobi algorithm" OFF)
option(BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)

set(GINKGO_CIRCULAR_DEPS_FLAGS "-Wl,--no-undefined")

if(BUILD_SHARED_LIBS AND (WIN32 OR CYGWIN) AND (GINKGO_BUILD_TESTS OR GINKGO_BUILD_EXAMPLES OR GINKGO_BUILD_BENCHMARKS))
    # Change shared libraries output only if this build has executable program with shared libraries.
    set(GINKGO_CHANGED_SHARED_LIBRARY TRUE)
    option(GINKGO_CHECK_PATH "Tell Ginkgo to check if the environment variable PATH is available for this build." ON)
    set(GINKGO_WINDOWS_SHARED_LIBRARY_RELPATH "windows_shared_library" CACHE STRING
        "Set Ginkgo's shared library relative path in windows. Current default is `windows_shared_library`. \
        This absolute path ${PROJECT_BINARY_DIR}/GINKGO_WINDOWS_SHARED_LIBRARY_RELPATH must be in the environment variable PATH.")
    set(GINKGO_WINDOWS_SHARED_LIBRARY_PATH ${PROJECT_BINARY_DIR}/${GINKGO_WINDOWS_SHARED_LIBRARY_RELPATH})
else()
    set(GINKGO_CHANGED_SHARED_LIBRARY FALSE)
endif()

if(GINKGO_BUILD_TESTS AND (GINKGO_BUILD_CUDA OR GINKGO_BUILD_OMP OR GINKGO_BUILD_HIP))
    message(STATUS "GINKGO_BUILD_TESTS is ON, enabling GINKGO_BUILD_REFERENCE")
    set(GINKGO_BUILD_REFERENCE ON CACHE BOOL "Compile reference CPU kernels" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if (BUILD_SHARED_LIBS)
    set(GINKGO_STATIC_OR_SHARED SHARED)
else()
    set(GINKGO_STATIC_OR_SHARED STATIC)
endif()

# Ensure we have a debug postfix
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()

if(GINKGO_BUILD_TESTS)
    # Configure CTest
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestCustom.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake @ONLY)

    # For testing, we need some special matrices
    add_subdirectory(matrices)

    enable_testing()
    include(CTest)
endif()

if (GINKGO_WITH_CLANG_TIDY)
  find_program(GINKGO_CLANG_TIDY_PATH clang-tidy)
endif()

if (GINKGO_WITH_IWYU)
  find_program(GINKGO_IWYU_PATH iwyu)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

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

set(GINKGO_HIP_PLATFORM_NVCC 0)
set(GINKGO_HIP_PLATFORM_HCC 0)

if (GINKGO_BUILD_HIP)
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
            set(ENV{HIP_PATH} ${HIP_PATH})
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()

    find_program(GINKGO_HIPCONFIG_PATH hipconfig HINTS "${HIP_PATH}/bin")
    if(GINKGO_HIPCONFIG_PATH)
        execute_process(COMMAND ${GINKGO_HIPCONFIG_PATH} --platform OUTPUT_VARIABLE GINKGO_HIP_PLATFORM)
    elseif(DEFINED ENV{HIP_PLATFORM})
        set(GINKGO_HIP_PLATFORM "$ENV{HIP_PLATFORM}")
    else()
        message(FATAL_ERROR "No platform could be found for HIP. "
            "Set and export the environment variable HIP_PLATFORM.")
    endif()
    message(STATUS "HIP platform set to ${GINKGO_HIP_PLATFORM}")

    if (GINKGO_HIP_PLATFORM STREQUAL "hcc")
        set(GINKGO_HIP_PLATFORM_HCC 1)
    elseif (GINKGO_HIP_PLATFORM STREQUAL "nvcc")
        set(GINKGO_HIP_PLATFORM_NVCC 1)
    endif()
endif()

configure_file(${Ginkgo_SOURCE_DIR}/include/ginkgo/config.hpp.in
    ${Ginkgo_BINARY_DIR}/include/ginkgo/config.hpp @ONLY)

# Load CMake helpers
include(cmake/build_helpers.cmake)
include(cmake/build_type_helpers.cmake)
include(cmake/hip_helpers.cmake)
include(cmake/install_helpers.cmake)

# This is modified from https://gitlab.kitware.com/cmake/community/wikis/FAQ#dynamic-replace
if(MSVC)
    if(BUILD_SHARED_LIBS)
        ginkgo_switch_to_windows_dynamic("CXX")
        ginkgo_switch_to_windows_dynamic("C")
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    else()
        ginkgo_switch_to_windows_static("CXX")
        ginkgo_switch_to_windows_static("C")
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)
    endif()
endif()

# Try to find the third party packages before using our subdirectories
include(cmake/package_helpers.cmake)
ginkgo_find_package(GTest "GTest::GTest;GTest::Main" FALSE 1.8.1)
ginkgo_find_package(gflags gflags FALSE 2.2.2)
ginkgo_find_package(RapidJSON rapidjson TRUE 1.1.0)
add_subdirectory(third_party)    # Third-party tools and libraries

# Ginkgo core libraries
# Needs to be first in order for `CMAKE_CUDA_DEVICE_LINK_EXECUTABLE` to be
# propagated to the other parts of Ginkgo in case of building as static libraries
if(GINKGO_BUILD_CUDA)
    add_subdirectory(cuda)       # High-performance kernels for NVIDIA GPUs
endif()
add_subdirectory(core)           # Core Ginkgo types and top-level functions
if (GINKGO_BUILD_REFERENCE)
    add_subdirectory(reference)  # Reference kernel implementations
endif()
if (GINKGO_BUILD_OMP)
    add_subdirectory(omp)        # High-performance omp kernels
endif()
# HIP needs to be last because it builds the GINKGO_RPATH_FOR_HIP variable
# which needs to know the `ginkgo` target.
if(GINKGO_BUILD_HIP)
    add_subdirectory(hip)        # High-performance kernels for AMD or NVIDIA GPUs
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
# when bash is present and the developer tools are enabled
find_program(BASH bash)
if(NOT "${BASH}" STREQUAL "BASH-NOTFOUND" AND GINKGO_DEVEL_TOOLS)
    add_custom_target(generate_ginkgo_header ALL
        COMMAND ${Ginkgo_SOURCE_DIR}/dev_tools/scripts/update_ginkgo_header.sh
        WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR})
endif()
unset(BASH CACHE)


# Installation
include(cmake/information_helpers.cmake)
ginkgo_interface_information()
ginkgo_git_information()

if(GINKGO_BUILD_DOC)
    add_subdirectory(doc)
endif()

configure_file(${Ginkgo_SOURCE_DIR}/cmake/ginkgo.pc.in
    ${Ginkgo_BINARY_DIR}/ginkgo.pc @ONLY)

ginkgo_install()

add_custom_target(test_install
    COMMAND ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} -H${Ginkgo_SOURCE_DIR}/test_install
        -B${Ginkgo_BINARY_DIR}/test_install
        -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_CONFIG_DIR}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    COMMAND ${CMAKE_COMMAND} --build ${Ginkgo_BINARY_DIR}/test_install
    COMMAND ${Ginkgo_BINARY_DIR}/test_install/test_install
    COMMENT "Running a test on the installed binaries. This requires running `(sudo) make install` first.")

# Setup CPack
set(CPACK_PACKAGE_DESCRIPTION_FILE "${Ginkgo_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${Ginkgo_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_ICON "${Ginkgo_SOURCE_DIR}/assets/logo.png")
set(CPACK_PACKAGE_CONTACT "ginkgo.library@gmail.com")
include(CPack)

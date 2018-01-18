if(WIN32)
    cmake_minimum_required(VERSION 3.9)
else()
    cmake_minimum_required(VERSION 3.8)
endif()

option(DEVEL_TOOLS "Add development tools to the build system" ON)
option(BUILD_TESTS "Generate build files for unit tests" ON)
option(BUILD_EXAMPLES "Build Ginkgo's examples" ON)
option(BUILD_REFERENCE "Compile reference CPU kernels" OFF)
option(BUILD_CPU "Compile kernels for CPUs" OFF)
option(BUILD_GPU "Compile kernels for NVIDIA GPUs" OFF)


include(cmake/create_test.cmake)


if(BUILD_GPU)
    project(Ginkgo LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD_REQUIRED true)
    set(CMAKE_CUDA_STANDARD 11)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
      set(CMAKE_CUDA_FLAGS "-Wno-deprecated-declarations
            --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS}"
        CACHE string "")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9)
      set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr
            ${CMAKE_CUDA_FLAGS}" CACHE string "")
    endif()
else()
    project(Ginkgo LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(third_party)    # Third-party tools and libraries
add_subdirectory(core)           # Core Ginkgo types and top-level functions
if (BUILD_REFERENCE)
    add_subdirectory(reference)  # Reference kernel implementations
endif()
if (BUILD_CPU)
    add_subdirectory(cpu)        # High-performance cpu kernels
endif()
if(BUILD_GPU)
    add_subdirectory(gpu)        # High-performance gpu kernels
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(DEVEL_TOOLS)
    set(LICENSE_SCRIPT ${CMAKE_SOURCE_DIR}/dev_tools/scripts/add_license.sh)

    add_custom_target(add_license
        COMMAND ${LICENSE_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_dependencies(format add_license)
endif()

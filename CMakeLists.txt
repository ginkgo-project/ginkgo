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
option(BUILD_DOC "Generate documentation" ON)
option(BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)
option(SET_CUDA_HOST_COMPILER "Propagate selected C++ compiler to CUDA" OFF)
option(SKIP_DEPENDENCY_UPDATE
       "Do not update dependencies each time the project is rebuilt" OFF)


include(cmake/create_test.cmake)


if(BUILD_GPU)
    project(Ginkgo LANGUAGES CXX CUDA)
    include(cmake/detect_gpu.cmake)
    set(CMAKE_CUDA_STANDARD_REQUIRED true)
    set(CMAKE_CUDA_STANDARD 11)
    if(SET_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_FLAGS
            "${CMAKE_CUDA_FLAGS} --compiler-bindir ${CMAKE_CXX_COMPILER}")
    endif()
    # set CUDA_ARCH_OPTION strings
    set(CUDA_ARCH_OPTION "Auto" CACHE STRING
        "Select target NVIDIA GPU achitecture.
         Use ';' to seperate list.
         Arch name: Kepler, Maxwell, Pascal, Volta.
         Specified BIN, BIN(PTX), (PTX).
         MaxPTX will select the max ptx")
    # setting nvcc arch flags
    ginkgo_select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
    message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS_EXTRA}")
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
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
<<<<<<< HEAD

if(BUILD_DOC)
	add_subdirectory(doc)
endif()
=======
>>>>>>> develop

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
option(SET_CUDA_HOST_COMPILER "Propagate selected C++ compiler to CUDA" OFF)
option(SKIP_DEPENDENCY_UPDATE
       "Do not update dependencies each time the project is rebuilt" OFF)


include(cmake/create_test.cmake)


set(known_gpu_archs "30;35;50;60;61")
function(detect_installed_gpus out_variable)
    # detect arch of computor
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)
    file(WRITE ${__cufile} ""
        "#include <cstdio>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
        "  if (count == 0) return -1;\n"
        "  for (int device = 0; device < count; ++device)\n"
        "  {\n"
        "    cudaDeviceProp prop;\n"
        "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
        "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
        "  }\n"
        "  return 0;\n"
        "}\n")
    execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "--run" "${__cufile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __cu_res OUTPUT_VARIABLE __cu_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    

    if (NOT(__cu_res EQUAL 0))
        message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
        set(${out_variable} ${known_gpu_archs} PARENT_SCOPE)
    else()
        set(${out_variable} ${__cu_out} PARENT_SCOPE)
    endif()
endfunction()


function(select_nvcc_arch_flags out_variable)
    # List of arch names
    set(__archs_names "Auto" "Kepler" "Maxwell" "Pascal" "Volta" "All" "Manual")
    set(__archs_name_default "Auto")
    if(NOT CMAKE_CROSSCOMPILING)
        set(__archs_name_default "Auto")
    endif()

    # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
    set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU achitecture.")
    set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
    # verify CUDA_ARCH_NAME value
    if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
        string(REPLACE ";" ", " __archs_names "${__archs_names}")
        message(FATAL_ERROR "Only ${__archs_names} architeture names are supported.")
    endif()

    # get the version of nvcc
    set(__nvfile ${PROJECT_BINARY_DIR}/detect_nvcc_version.cu)
    file(WRITE ${__nvfile} ""
        "#include <cstdio>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
        "  if (count == 0) return -1;\n"
        "  std::printf(\"%d\", __CUDACC_VER_MAJOR__);\n"
        "  return 0;\n"
        "}\n")
    execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "--run" "${__nvfile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __nv_res OUTPUT_VARIABLE __nv_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if((__nv_res EQUAL 0) AND (__nv_out GREATER_EQUAL 9))
        list(APPEND known_gpu_archs "70")
    endif()

    if(${CUDA_ARCH_NAME} STREQUAL "Manual")
        set(CUDA_ARCH_BIN ${known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries")
    else()
        unset(CUDA_ARCH_BIN CACHE)
    endif()


    if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
        set(__cuda_arch_bin "30 35")
    elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
        set(__cuda_arch_bin "50")
    elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
        set(__cuda_arch_bin "60 61")
    elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
        set(__cuda_arch_bin "70")
    elseif(${CUDA_ARCH_NAME} STREQUAL "All")
        set(__cuda_arch_bin ${known_gpu_archs})
    elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
        detect_installed_gpus(__cuda_arch_bin)
    else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
        set(__cuda_arch_bin ${CUDA_ARCH_BIN})
    endif()

    # remove dots and convert to lists
    string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
    string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
    list(REMOVE_DUPLICATES __cuda_arch_bin)
    list(SORT __cuda_arch_bin)
    set(__nvcc_flags "")
    set(__nvcc_archs_readable "")

    # Tell NVCC to add binaries for the specified GPUs
    foreach(__arch ${__cuda_arch_bin})
        if(__arch LESS 30)
            message(FATAL_ERROR "Gingko Library does not support Fermi arch")
        elseif((__nv_out LESS 9) AND (__arch GREATER_EQUAL 70))
            message(FATAL_ERROR "Volta: Needs cuda version > 9.0")
        endif()
        list(APPEND __nvcc_flags "-gencode=arch=compute_${__arch},code=sm_${__arch}")
        list(APPEND __nvcc_archs_readable sm_${__arch})
    endforeach()
    set(__largest_arch "")
    list(GET __cuda_arch_bin -1 __largest_arch)
    list(APPEND __nvcc_flags "-gencode=arch=compute_${__largest_arch},code=compute_${__largest_arch}")
    string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
    set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
    set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()


if(BUILD_GPU)
    project(Ginkgo LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD_REQUIRED true)
    set(CMAKE_CUDA_STANDARD 11)
    if(SET_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_FLAGS
            "--compiler-bindir ${CMAKE_CXX_COMPILER} ${CMAKE_CUDA_FLAGS}")
    endif()
    select_nvcc_arch_flags(NVCC_ARCH_FLAGS)
    message(STATUS "Support Nvidia GPU Arch ${NVCC_ARCH_FLAGS_readable}")
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

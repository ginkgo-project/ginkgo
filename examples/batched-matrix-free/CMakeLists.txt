cmake_minimum_required(VERSION 3.21)
project(batched-solver LANGUAGES CXX)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif ()

include(CheckLanguage)
check_language(CUDA)
check_language(HIP)

set(language CXX)
set(compile_defs "DISABLE_GPU")
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(compile_defs "ENABLE_CUDA")
    set(language "CUDA")
elseif (CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(compile_defs "ENABLE_HIP")
    set(language "HIP")
endif ()

add_executable(batched-matrix-free batched-matrix-free.cpp apply.cpu.cpp apply.gpu.cpp)
set_target_properties(batched-matrix-free PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(batched-matrix-free Ginkgo::ginkgo)

set_source_files_properties(apply.gpu.cpp PROPERTIES COMPILE_DEFINITIONS ${compile_defs})
set_source_files_properties(apply.gpu.cpp PROPERTIES LANGUAGE ${language})
set_target_properties(batched-matrix-free PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

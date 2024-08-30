cmake_minimum_required(VERSION 3.21)
project(batched-solver LANGUAGES CXX)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif ()

include(FetchContent)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v3.2.0
)
FetchContent_MakeAvailable(cxxopts)

set(language CXX)
set(compile_defs "DISABLE_GPU")
set(apply_src apply.cpu.cpp)
if (GINKGO_BUILD_CUDA)
    enable_language(CUDA)
    set(compile_defs "ENABLE_CUDA")
    set(language "CUDA")
    list(APPEND apply_src apply.gpu.cpp)
elseif (GINKGO_BUILD_HIP)
    enable_language(HIP)
    set(compile_defs "ENABLE_HIP")
    set(language "HIP")
    list(APPEND apply_src apply.gpu.cpp)
else ()
    list(APPEND apply_src apply.not-impl.cpp)
endif ()

find_package(OpenMP REQUIRED)

add_executable(batched-matrix-free batched-matrix-free.cpp ${apply_src})
set_target_properties(batched-matrix-free PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(batched-matrix-free Ginkgo::ginkgo cxxopts::cxxopts OpenMP::OpenMP_CXX)

set_source_files_properties(apply.gpu.cpp PROPERTIES COMPILE_DEFINITIONS ${compile_defs})
set_source_files_properties(apply.gpu.cpp PROPERTIES LANGUAGE ${language})
set_target_properties(batched-matrix-free PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

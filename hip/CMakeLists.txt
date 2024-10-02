cmake_minimum_required(VERSION 3.21)
include(${PROJECT_SOURCE_DIR}/cmake/template_instantiation.cmake)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip matrix/csr_kernels.instantiate.cpp CSR_INSTANTIATE)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip matrix/fbcsr_kernels.instantiate.cpp FBCSR_INSTANTIATE)
# we don't split up the dense kernels into distinct compilations
list(APPEND GKO_UNIFIED_COMMON_SOURCES ${PROJECT_SOURCE_DIR}/common/unified/matrix/dense_kernels.instantiate.cpp)
set(GINKGO_HIP_SOURCES
    base/device.hip.cpp
    base/exception.hip.cpp
    base/executor.hip.cpp
    base/memory.hip.cpp
    base/roctx.hip.cpp
    base/scoped_device_id.hip.cpp
    base/stream.hip.cpp
    base/timer.hip.cpp
    base/version.hip.cpp
    ${CSR_INSTANTIATE}
    ${FBCSR_INSTANTIATE}
    preconditioner/batch_jacobi_kernels.hip.cpp
    solver/batch_bicgstab_kernels.hip.cpp
    solver/batch_cg_kernels.hip.cpp
    solver/lower_trs_kernels.hip.cpp
    solver/upper_trs_kernels.hip.cpp
    ${GKO_UNIFIED_COMMON_SOURCES}
    ${GKO_CUDA_HIP_COMMON_SOURCES}
    )
if(GINKGO_BENCHMARK_ENABLE_TUNING)
    list(APPEND GINKGO_HIP_SOURCES ${Ginkgo_SOURCE_DIR}/benchmark/utils/tuning_variables.cpp)
endif()

if(hipfft_FOUND)
    list(APPEND GINKGO_HIP_SOURCES matrix/fft_kernels.hip.cpp)
else()
    list(APPEND GINKGO_HIP_SOURCES matrix/fft_kernels_stub.hip.cpp)
endif()

set(GKO_HIP_JACOBI_MAX_BLOCK_SIZE 64)
if(GINKGO_JACOBI_FULL_OPTIMIZATIONS)
    set(GKO_HIP_JACOBI_BLOCK_SIZES)
    foreach(blocksize RANGE 1 ${GKO_HIP_JACOBI_MAX_BLOCK_SIZE})
        list(APPEND GKO_HIP_JACOBI_BLOCK_SIZES ${blocksize})
    endforeach()
else()
    set(GKO_HIP_JACOBI_BLOCK_SIZES 1 2 4 8 13 16 32 ${GKO_HIP_JACOBI_MAX_BLOCK_SIZE})
    list(REMOVE_DUPLICATES GKO_HIP_JACOBI_BLOCK_SIZES)
endif()
jacobi_generated_files(GKO_HIP_JACOBI_SOURCES "${GKO_HIP_JACOBI_BLOCK_SIZES}")
# The 3D indexing used in Jacobi kernel triggers an instruction selection bug in Debug builds
# Probably the same as https://github.com/llvm/llvm-project/issues/67574
# Fixed in ROCm 6.0 https://github.com/ROCm/llvm-project/commit/cd7f574a1fd1d3f3e8b9c1cae61fa8133a51de5f
# and in LLVM trunk https://github.com/llvm/llvm-project/commit/cc3d2533cc2e4ea06981b86ede5087fbf801e789
set_source_files_properties(${GKO_HIP_JACOBI_SOURCES} PROPERTIES COMPILE_OPTIONS $<$<CONFIG:Debug>:-O2>)
list(APPEND GINKGO_HIP_SOURCES ${GKO_HIP_JACOBI_SOURCES})
string(REPLACE ";" "," GKO_JACOBI_BLOCK_SIZES_CODE "${GKO_HIP_JACOBI_BLOCK_SIZES}")
configure_file(${Ginkgo_SOURCE_DIR}/common/cuda_hip/preconditioner/jacobi_common.hpp.in common/cuda_hip/preconditioner/jacobi_common.hpp)

set_source_files_properties(${GINKGO_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(ginkgo_hip $<TARGET_OBJECTS:ginkgo_hip_device> ${GINKGO_HIP_SOURCES})

target_include_directories(ginkgo_hip
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} # for generated headers like jacobi_common.hip.hpp
        )
target_compile_definitions(ginkgo_hip PRIVATE GKO_COMPILING_HIP GKO_DEVICE_NAMESPACE=hip)

target_link_libraries(ginkgo_hip PUBLIC ginkgo_device)
target_link_libraries(ginkgo_hip PRIVATE hip::host roc::hipblas roc::hipsparse hip::hiprand roc::rocrand roc::rocthrust)
if (hipfft_FOUND)
    target_link_libraries(ginkgo_hip PRIVATE hip::hipfft)
endif()
if (GINKGO_HAVE_ROCTX)
    target_link_libraries(ginkgo_hip PRIVATE roc::roctx)
endif()

target_compile_options(ginkgo_hip PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)


ginkgo_compile_features(ginkgo_hip)
ginkgo_default_includes(ginkgo_hip)
ginkgo_install_library(ginkgo_hip)

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_hip "GKO_COMPILING_HIP;GKO_DEVICE_NAMESPACE=hip")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

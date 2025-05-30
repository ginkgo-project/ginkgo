cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
function(split_list_average input_list num_parts)
    message("${input_list}")
    message("${${input_list}}")
    list(LENGTH ${input_list} total_length)
    message("${input_list} contains ${total_length}")
    message("generate total ${num_parts}")
    foreach(i RANGE 1 ${num_parts})
        message("generate ${i}")
        math(EXPR start "${total_length} * (${i} - 1) / ${num_parts}")
        math(EXPR end "${total_length} * ${i} / ${num_parts} - 1")
        message("${start} - ${end}")
        set(part "")
        foreach(j RANGE ${start} ${end})
            list(GET ${input_list} ${j} item)
            list(APPEND part ${item})
        endforeach()
        set(part_list_var ${input_list}_${i})
        set(${part_list_var} ${part} PARENT_SCOPE)
    endforeach()
endfunction()

function(configure_cuda target source_list)
    target_sources(${target} PRIVATE ${${source_list}})
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        # remove false positive CUDA warnings when calling one<T>() and zero<T>()
        # and allows the usage of std::array for nvidia GPUs
        target_compile_options(
            ${target}
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        if(MSVC)
            target_compile_options(
                ${target}
                PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            )
        else()
            target_compile_options(
                ${target}
                PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
            )
        endif()
    endif()

    ginkgo_compile_features(${target})
    target_compile_definitions(
        ${target}
        PRIVATE GKO_COMPILING_CUDA GKO_DEVICE_NAMESPACE=cuda
    )
    if(GINKGO_CUDA_CUSTOM_THRUST_NAMESPACE)
        target_compile_definitions(
            ${target}
            PRIVATE THRUST_CUB_WRAPPED_NAMESPACE=gko
        )
    endif()

    # include path for generated headers like jacobi_common.hpp
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(
        ${target}
        PRIVATE
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
            CUDA::cufft
            nvtx::nvtx
    )
    # NVTX3 is header-only and requires dlopen/dlclose in static builds
    target_link_libraries(${target} PUBLIC ginkgo_device ${CMAKE_DL_LIBS})

    ginkgo_default_includes(${target})
    ginkgo_install_library(${target})
endfunction()

include(${PROJECT_SOURCE_DIR}/cmake/template_instantiation.cmake)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip matrix/csr_kernels.instantiate.cpp CSR_INSTANTIATE)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip matrix/fbcsr_kernels.instantiate.cpp FBCSR_INSTANTIATE)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip solver/batch_bicgstab_launch.instantiate.cpp BATCH_BICGSTAB_INSTANTIATE1)
add_instantiation_files(. solver/batch_bicgstab_launch.instantiate.cu BATCH_BICGSTAB_INSTANTIATE2)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip solver/batch_cg_launch.instantiate.cpp BATCH_CG_INSTANTIATE1)
add_instantiation_files(. solver/batch_cg_launch.instantiate.cu BATCH_CG_INSTANTIATE2)
# we don't split up the dense kernels into distinct compilations
list(
    APPEND
    GKO_UNIFIED_COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/common/unified/matrix/dense_kernels.instantiate.cpp
)
if(GINKGO_JACOBI_FULL_OPTIMIZATIONS)
    set(GKO_CUDA_JACOBI_BLOCK_SIZES)
    foreach(blocksize RANGE 1 32)
        list(APPEND GKO_CUDA_JACOBI_BLOCK_SIZES ${blocksize})
    endforeach()
else()
    set(GKO_CUDA_JACOBI_BLOCK_SIZES
        1
        2
        4
        8
        13
        16
        32
    )
endif()
jacobi_generated_files(GKO_CUDA_JACOBI_SOURCES "${GKO_CUDA_JACOBI_BLOCK_SIZES}")
# override the default language mapping for the common files, set them to CUDA
foreach(
    source_file
    IN
    LISTS
        GKO_UNIFIED_COMMON_SOURCES
        GKO_CUDA_HIP_COMMON_SOURCES
        GKO_CUDA_JACOBI_SOURCES
        CSR_INSTANTIATE
        FBCSR_INSTANTIATE
        BATCH_BICGSTAB_INSTANTIATE1
        BATCH_BICGSTAB_INSTANTIATE2
        BATCH_CG_INSTANTIATE1
        BATCH_CG_INSTANTIATE2
)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
endforeach(source_file)
set(CUDA_SOURCES
    base/device.cpp
    base/exception.cpp
    base/executor.cpp
    base/memory.cpp
    base/nvtx.cpp
    base/scoped_device_id.cpp
    base/stream.cpp
    base/timer.cpp
    base/version.cpp
    ${CSR_INSTANTIATE}
    ${FBCSR_INSTANTIATE}
    matrix/fft_kernels.cu
    preconditioner/batch_jacobi_kernels.cu
    solver/batch_bicgstab_kernels.cu
    ${BATCH_BICGSTAB_INSTANTIATE1}
    ${BATCH_BICGSTAB_INSTANTIATE2}
    solver/batch_cg_kernels.cu
    ${BATCH_CG_INSTANTIATE1}
    ${BATCH_CG_INSTANTIATE2}
    solver/lower_trs_kernels.cu
    solver/upper_trs_kernels.cu
    ${GKO_UNIFIED_COMMON_SOURCES}
    ${GKO_CUDA_HIP_COMMON_SOURCES}
    ${GKO_CUDA_JACOBI_SOURCES}
)
# We still configure it ginkgo_cuda library not interface library when spliting library to reduce the effort in pkgconfig
add_library(ginkgo_cuda $<TARGET_OBJECTS:ginkgo_cuda_device> "")
if(${GINKGO_SPLIT_LIBRARY_NUM} GREATER 1)
    split_list_average(CUDA_SOURCES ${GINKGO_SPLIT_LIBRARY_NUM})
    foreach(i RANGE 1 ${GINKGO_SPLIT_LIBRARY_NUM})
        list(LENGTH CUDA_SOURCES_${i} part_length)
        message(
            "CUDA_SOURCES_${i} contains ${part_length}: ${CUDA_SOURCES_${i}}"
        )
    endforeach()
    foreach(i RANGE 1 ${GINKGO_SPLIT_LIBRARY_NUM})
        add_library(ginkgo_cuda_${i} "")
        configure_cuda(ginkgo_cuda_${i} CUDA_SOURCES_${i})
        target_link_libraries(ginkgo_cuda INTERFACE ginkgo_cuda_${i})
    endforeach()
    ginkgo_compile_features(ginkgo_cuda)
    ginkgo_default_includes(ginkgo_cuda)
    ginkgo_install_library(ginkgo_cuda)
else()
    # keep the original behavior to avoid different name between target and library when num == 1
    configure_cuda(ginkgo_cuda CUDA_SOURCES)
endif()

string(
    REPLACE
    ";"
    ","
    GKO_JACOBI_BLOCK_SIZES_CODE
    "${GKO_CUDA_JACOBI_BLOCK_SIZES}"
)
configure_file(
    ${Ginkgo_SOURCE_DIR}/common/cuda_hip/preconditioner/jacobi_common.hpp.in
    common/cuda_hip/preconditioner/jacobi_common.hpp
)

if(GINKGO_CHECK_CIRCULAR_DEPS)
    set(check_header_def "GKO_COMPILING_CUDA;GKO_DEVICE_NAMESPACE=cuda")
    if(GINKGO_CUDA_CUSTOM_THRUST_NAMESPACE)
        set(check_header_def
            "${check_header_def};THRUST_CUB_WRAPPED_NAMESPACE=gko"
        )
    endif()
    ginkgo_check_headers(ginkgo_cuda "${check_header_def}")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

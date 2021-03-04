
include(GenerateExportHeader)

if(NOT GINKGO_BUILD_CUDA)
    add_library(ginkgo_cuda
        $<TARGET_OBJECTS:ginkgo_cuda_device>
        cuda_hooks.cpp)
    target_link_libraries(ginkgo_cuda PUBLIC ginkgo_core_device)
    ginkgo_compile_features(ginkgo_cuda)
    ginkgo_default_includes(ginkgo_cuda)
    ginkgo_install_library(ginkgo_cuda cuda)
    generate_export_header(ginkgo_cuda
        EXPORT_FILE_NAME "${Ginkgo_BINARY_DIR}/include/ginkgo/export_cuda.hpp"
        BASE_NAME "GKO_CUDA"
        INCLUDE_GUARD_NAME "GKO_PUBLIC_API_EXPORT_CUDA_HPP_")
endif()

if (NOT GINKGO_BUILD_DPCPP)
    add_library(ginkgo_dpcpp
        $<TARGET_OBJECTS:ginkgo_dpcpp_device>
        dpcpp_hooks.cpp)
    target_link_libraries(ginkgo_dpcpp PUBLIC ginkgo_core_device)
    ginkgo_compile_features(ginkgo_dpcpp)
    ginkgo_default_includes(ginkgo_dpcpp)
    ginkgo_install_library(ginkgo_dpcpp dpcpp)
    generate_export_header(ginkgo_dpcpp
        EXPORT_FILE_NAME "${Ginkgo_BINARY_DIR}/include/ginkgo/export_dpcpp.hpp"
        BASE_NAME "GKO_DPCPP"
        INCLUDE_GUARD_NAME "GKO_PUBLIC_API_EXPORT_DPCPP_HPP_")
endif()

if(NOT GINKGO_BUILD_HIP)
    add_library(ginkgo_hip
        $<TARGET_OBJECTS:ginkgo_hip_device>
        hip_hooks.cpp)
    target_link_libraries(ginkgo_hip PUBLIC ginkgo_core_device)
    ginkgo_compile_features(ginkgo_hip)
    ginkgo_default_includes(ginkgo_hip)
    ginkgo_install_library(ginkgo_hip hip)
    generate_export_header(ginkgo_hip
        EXPORT_FILE_NAME "${Ginkgo_BINARY_DIR}/include/ginkgo/export_hip.hpp"
        BASE_NAME "GKO_HIP"
        INCLUDE_GUARD_NAME "GKO_PUBLIC_API_EXPORT_HIP_HPP_")
endif()

if (NOT GINKGO_BUILD_OMP)
    add_library(ginkgo_omp
        $<TARGET_OBJECTS:ginkgo_omp_device>
        omp_hooks.cpp)
    ginkgo_compile_features(ginkgo_omp)
    target_link_libraries(ginkgo_omp PUBLIC ginkgo_core_device ginkgo_cuda)
    target_link_libraries(ginkgo_omp PUBLIC ginkgo_hip)
    target_link_libraries(ginkgo_omp PUBLIC ginkgo_dpcpp)
    ginkgo_default_includes(ginkgo_omp)
    ginkgo_install_library(ginkgo_omp omp)
    generate_export_header(ginkgo_omp
        EXPORT_FILE_NAME "${Ginkgo_BINARY_DIR}/include/ginkgo/export_omp.hpp"
        BASE_NAME "GKO_OMP"
        INCLUDE_GUARD_NAME "GKO_PUBLIC_API_EXPORT_OMP_HPP_")
endif()

if (NOT GINKGO_BUILD_REFERENCE)
    add_library(ginkgo_reference
        $<TARGET_OBJECTS:ginkgo_reference_device>
        reference_hooks.cpp)
    target_link_libraries(ginkgo_reference PUBLIC ginkgo_core_device)
    ginkgo_compile_features(ginkgo_reference)
    ginkgo_default_includes(ginkgo_reference)
    ginkgo_install_library(ginkgo_reference reference)
    generate_export_header(ginkgo_reference
        EXPORT_FILE_NAME "${Ginkgo_BINARY_DIR}/include/ginkgo/export_reference.hpp"
        BASE_NAME "GKO_REFERENCE"
        INCLUDE_GUARD_NAME "GKO_PUBLIC_API_EXPORT_REFERENCE_HPP_")
endif()

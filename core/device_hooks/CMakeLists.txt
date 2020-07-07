if(NOT GINKGO_BUILD_CUDA)
    add_library(ginkgo_cuda
        $<TARGET_OBJECTS:ginkgo_cuda_device>
        cuda_hooks.cpp)
    ginkgo_compile_features(ginkgo_cuda)
    target_link_libraries(ginkgo_cuda PUBLIC ginkgo_hip)
    ginkgo_default_includes(ginkgo_cuda)
    ginkgo_install_library(ginkgo_cuda cuda)
endif()

if (NOT GINKGO_BUILD_OMP)
    add_library(ginkgo_omp
        $<TARGET_OBJECTS:ginkgo_omp_device>
        omp_hooks.cpp)
    ginkgo_compile_features(ginkgo_omp)
    target_link_libraries(ginkgo_omp PUBLIC ginkgo_cuda)
    target_link_libraries(ginkgo_omp PUBLIC ginkgo_hip)
    ginkgo_default_includes(ginkgo_omp)
    ginkgo_install_library(ginkgo_omp omp)
endif()

if (NOT GINKGO_BUILD_REFERENCE)
    add_library(ginkgo_reference
        $<TARGET_OBJECTS:ginkgo_reference_device>
        reference_hooks.cpp)
    ginkgo_compile_features(ginkgo_reference)
    ginkgo_default_includes(ginkgo_reference)
    ginkgo_install_library(ginkgo_reference reference)
endif()

if(NOT GINKGO_BUILD_HIP)
    add_library(ginkgo_hip
        $<TARGET_OBJECTS:ginkgo_hip_device>
        hip_hooks.cpp)
    ginkgo_compile_features(ginkgo_hip)
    ginkgo_default_includes(ginkgo_hip)
    ginkgo_install_library(ginkgo_hip hip)
endif()

if(NOT BUILD_GPU)
    add_library(ginkgo_gpu
        $<TARGET_OBJECTS:ginkgo_gpu_device>
        gpu_hooks.cpp)
    ginkgo_default_includes(ginkgo_gpu)
    ginkgo_install_library(ginkgo_gpu gpu)
endif()

if (NOT BUILD_OMP)
    add_library(ginkgo_omp
        $<TARGET_OBJECTS:ginkgo_omp_device>
        omp_hooks.cpp)
    target_link_libraries(ginkgo_omp PUBLIC ginkgo_gpu)
    ginkgo_default_includes(ginkgo_omp)
    ginkgo_install_library(ginkgo_omp omp)
endif()

if (NOT BUILD_REFERENCE)
    add_library(ginkgo_reference
        $<TARGET_OBJECTS:ginkgo_reference_device>
        reference_hooks.cpp)
    ginkgo_default_includes(ginkgo_reference)
    ginkgo_install_library(ginkgo_reference reference)
endif()

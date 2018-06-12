if(NOT BUILD_GPU)
    add_library(ginkgo_gpu
        $<TARGET_OBJECTS:ginkgo_gpu_device>
        gpu_hooks.cpp)
    ginkgo_default_includes(ginkgo_gpu)
    ginkgo_install_library(ginkgo_gpu gpu)
endif()

if (NOT BUILD_CPU)
    add_library(ginkgo_cpu
        $<TARGET_OBJECTS:ginkgo_cpu_device>
        cpu_hooks.cpp)
    target_link_libraries(ginkgo_cpu PUBLIC ginkgo_gpu)
    ginkgo_default_includes(ginkgo_cpu)
    ginkgo_install_library(ginkgo_cpu cpu)
endif()

if (NOT BUILD_REFERENCE)
    add_library(ginkgo_reference
        $<TARGET_OBJECTS:ginkgo_reference_device>
        reference_hooks.cpp)
    ginkgo_default_includes(ginkgo_reference)
    ginkgo_install_library(ginkgo_reference reference)
endif()

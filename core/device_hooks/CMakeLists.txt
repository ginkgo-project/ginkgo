if(NOT BUILD_GPU)
    add_library(ginkgo_gpu
        $<TARGET_OBJECTS:ginkgo_gpu_device>
        gpu_hooks.cpp)
    target_include_directories(ginkgo_gpu PUBLIC ${PROJECT_SOURCE_DIR})
endif()

if (NOT BUILD_CPU)
    add_library(ginkgo_cpu
        $<TARGET_OBJECTS:ginkgo_cpu_device>
        cpu_hooks.cpp)
    target_include_directories(ginkgo_cpu PUBLIC ${PROJECT_SOURCE_DIR})
    target_link_libraries(ginkgo_cpu PUBLIC ginkgo_gpu)
endif()

if (NOT BUILD_REFERENCE)
    add_library(ginkgo_reference
        $<TARGET_OBJECTS:ginkgo_reference_device>
        reference_hooks.cpp)
    target_include_directories(ginkgo_reference PUBLIC ${PROJECT_SOURCE_DIR})
endif()

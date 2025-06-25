ginkgo_add_object_library(ginkgo_hip_device
    executor.cpp
)
if(GINKGO_BUILD_HIP)
    target_compile_definitions(
        ginkgo_hip_device
        PRIVATE GKO_COMPILING_HIP_DEVICE
    )
endif()

ginkgo_add_object_library(ginkgo_cuda_device executor.cpp)
if(GINKGO_BUILD_CUDA)
    target_compile_definitions(
        ginkgo_cuda_device
        PRIVATE GKO_COMPILING_CUDA_DEVICE
    )
endif()

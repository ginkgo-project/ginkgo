ginkgo_create_hip_test(hip_executor)
ginkgo_create_test(index_set RESOURCE_TYPE hipgpu)
ginkgo_create_hip_test(kernel_launch)
ginkgo_create_test(lin_op RESOURCE_TYPE hipgpu)
ginkgo_create_hip_test(math)
ginkgo_create_test(memory EXECUTABLE_NAME memory_test RESOURCE_TYPE hipgpu)
set(additional_libs)
if(hipfft_FOUND)
    set(additional_libs hip::hipfft)
endif()
ginkgo_create_hip_test(
    exception_helpers
    ADDITIONAL_LIBRARIES
    roc::hipblas
    roc::hipsparse
    hip::hiprand
    roc::rocrand
    ${additional_libs}
)
ginkgo_create_hip_test(scoped_device_id)

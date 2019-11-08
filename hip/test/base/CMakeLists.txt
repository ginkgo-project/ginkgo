ginkgo_create_hip_test(hip_executor)
ginkgo_create_hip_test(math)
ginkgo_create_hip_test(hip_memory_space)
# Only hcc needs the libraries. nvcc only requires the headers.
if (GINKGO_HIP_PLATFORM MATCHES "hcc")
    ginkgo_create_hip_test(exception_helpers roc::hipblas roc::hipsparse)
else()
    ginkgo_create_hip_test(exception_helpers)
endif()

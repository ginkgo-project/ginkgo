ginkgo_create_hip_test(hip_executor)
# hipblas and hipsparse are indeed required, but somehow should only be passed
# under the `hcc` platform.
if (GINKGO_HIP_PLATFORM MATCHES "hcc")
    ginkgo_create_hip_test(exception_helpers roc::hipblas roc::hipsparse)
else()
    ginkgo_create_hip_test(exception_helpers)
endif()

ginkgo_create_test(fbcsr_kernels RESOURCE_TYPE hipgpu)
if (hipfft_FOUND)
    ginkgo_create_hip_test(fft_kernels)
endif()

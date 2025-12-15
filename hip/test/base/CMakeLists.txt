ginkgo_create_hip_test(hip_executor)
ginkgo_create_test(index_set RESOURCE_TYPE hipgpu)
if(GINKGO_HAVE_HWLOC)
    find_package(NUMA)
    if(NUMA_FOUND)
        ginkgo_create_hip_test(
            hip_executor_topology
            ADDITIONAL_LIBRARIES
            NUMA::NUMA
        )
    else()
        message(STATUS "NUMA not found, disabling HWLOC test.")
    endif()
endif()
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

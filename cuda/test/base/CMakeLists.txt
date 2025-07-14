ginkgo_create_test(array EXECUTABLE_NAME array_test RESOURCE_TYPE cudagpu) # array collides with C++ stdlib header
ginkgo_create_cuda_test(cuda_executor)
ginkgo_create_test(index_set RESOURCE_TYPE cudagpu)
if(GINKGO_HAVE_HWLOC)
    find_package(NUMA)
    if(NUMA_FOUND)
        ginkgo_create_cuda_test(
            cuda_executor_topology
            ADDITIONAL_LIBRARIES
            NUMA::NUMA
        )
    else()
        message(STATUS "NUMA not found, disabling HWLOC test.")
    endif()
endif()
ginkgo_create_cuda_test(exception_helpers)
ginkgo_create_cuda_test(kernel_launch)
ginkgo_create_test(lin_op RESOURCE_TYPE cudagpu)
ginkgo_create_cuda_test(math)
ginkgo_create_test(memory EXECUTABLE_NAME memory_test RESOURCE_TYPE cudagpu)
ginkgo_create_cuda_test(scoped_device_id)

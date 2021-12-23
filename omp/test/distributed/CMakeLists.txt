ginkgo_create_test(partition_kernels)
if(GINKGO_BUILD_MPI)
    ginkgo_create_test(matrix_kernels)
endif()

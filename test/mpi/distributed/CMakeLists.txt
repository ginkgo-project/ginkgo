ginkgo_create_common_and_reference_test(assembly MPI_SIZE 3 LABELS distributed)
ginkgo_create_common_and_reference_test(matrix MPI_SIZE 3 LABELS distributed)
ginkgo_create_common_and_reference_test(
    partition_helpers
    MPI_SIZE
    3
    LABELS
    distributed
)
ginkgo_create_common_and_reference_test(spgemm MPI_SIZE 3 LABELS distributed)
if(GINKGO_BUILD_CUDA)
    target_compile_definitions(
        test_mpi_distributed_spgemm_cuda
        PRIVATE GKO_CUDA_TOOLKIT_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
    )
endif()
ginkgo_create_common_and_reference_test(vector MPI_SIZE 3 LABELS distributed)
# reduce the number of OpenMP threads per MPI rank to 2, so that in total 12 cores are used
ginkgo_create_common_and_reference_test(
    row_gatherer
    MPI_SIZE
    6
    LABELS
    distributed
    RESOURCE_LOCAL_CORES
    2
)

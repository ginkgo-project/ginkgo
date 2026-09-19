ginkgo_create_common_and_reference_test(
    pgm
    MPI_SIZE
    3
    DISABLE_EXECUTORS
    dpcpp
    LABELS
    distributed
)

ginkgo_create_common_and_reference_test(
    uniform_coarsening
    MPI_SIZE
    3
    DISABLE_EXECUTORS
    dpcpp
    LABELS
    distributed
)

ginkgo_create_common_and_reference_test(
    pmis
    MPI_SIZE
    3
    DISABLE_EXECUTORS
    dpcpp
    LABELS
    distributed
)

if(GINKGO_BUILD_CUDA)
    target_compile_definitions(
        test_mpi_multigrid_pmis_cuda
        PRIVATE GKO_CUDA_TOOLKIT_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
    )
endif()

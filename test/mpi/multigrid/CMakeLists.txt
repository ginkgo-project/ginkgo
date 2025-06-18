ginkgo_create_common_and_reference_test(
    pgm
    MPI_SIZE
    3
    DISABLE_EXECUTORS
    dpcpp
    LABELS
    distributed
)
ginkgo_create_common_and_reference_test(fixed_coarsening MPI_SIZE 3 DISABLE_EXECUTORS dpcpp LABELS distributed)

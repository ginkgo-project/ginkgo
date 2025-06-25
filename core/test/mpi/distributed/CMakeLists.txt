ginkgo_create_test(helpers MPI_SIZE 1 LABELS distributed)
ginkgo_create_test(matrix MPI_SIZE 1 LABELS distributed)
ginkgo_create_test(collective_communicator MPI_SIZE 6 LABELS distributed)
ginkgo_create_test(row_gatherer MPI_SIZE 6 LABELS distributed)
ginkgo_create_test(vector_cache MPI_SIZE 3 LABELS distributed)

add_subdirectory(preconditioner)
add_subdirectory(solver)

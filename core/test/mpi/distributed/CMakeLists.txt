ginkgo_create_test(helpers MPI_SIZE 1)
ginkgo_create_test(matrix MPI_SIZE 1)
ginkgo_create_test(dense_communicator MPI_SIZE 6)

add_subdirectory(preconditioner)
add_subdirectory(solver)

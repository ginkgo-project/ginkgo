ginkgo_create_test(helpers MPI_SIZE 1)
ginkgo_create_test(matrix MPI_SIZE 1)
ginkgo_create_test(vector_cache MPI_SIZE 3)

add_subdirectory(preconditioner)
add_subdirectory(solver)

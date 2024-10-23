ginkgo_create_test(helpers MPI_SIZE 1)
ginkgo_create_test(matrix MPI_SIZE 1)
ginkgo_create_test(dense_communicator MPI_SIZE 6)
ginkgo_create_test(row_gatherer MPI_SIZE 6)

if(NOT GINKGO_HAVE_OPENMPI_PRE_4_1_X)
    ginkgo_create_test(neighborhood_communicator MPI_SIZE 6)
endif()

add_subdirectory(preconditioner)
add_subdirectory(solver)

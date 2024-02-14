ginkgo_create_test(helpers MPI_SIZE 1)
ginkgo_create_test(matrix MPI_SIZE 1)
if (GINKGO_HAVE_CXX17)
    ginkgo_create_test(sparse_communicator MPI_SIZE 6)
endif ()

add_subdirectory(preconditioner)

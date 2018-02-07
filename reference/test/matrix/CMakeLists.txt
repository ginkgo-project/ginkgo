create_test(csr_kernels)
create_test(dense_kernels)
create_test(ell_kernels)
create_test(sliced_ell_kernels)

configure_file(data/dense_real_square.mtx data/dense_real_square.mtx COPYONLY)

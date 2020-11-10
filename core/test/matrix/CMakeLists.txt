ginkgo_create_test(coo)
ginkgo_create_test(coo_builder)
ginkgo_create_test(csr)
ginkgo_create_test(csr_builder)
ginkgo_create_test(dense)
ginkgo_create_test(diagonal)
ginkgo_create_test(ell)
add_library(test_fbcsr_sample fbcsr_sample.cpp)
target_include_directories(test_fbcsr_sample
  PRIVATE
  "$<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}/include>"
  )
ginkgo_create_test(fbcsr test_fbcsr_sample)
ginkgo_create_test(fbcsr_builder)
ginkgo_create_test(hybrid)
ginkgo_create_test(identity)
ginkgo_create_test(permutation)
ginkgo_create_test(sellp)
ginkgo_create_test(sparsity_csr)

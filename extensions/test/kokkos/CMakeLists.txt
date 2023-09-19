find_package(Kokkos 4.1 REQUIRED)

ginkgo_create_test(types ADDITIONAL_LIBRARIES Kokkos::kokkos)

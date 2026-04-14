include(${PROJECT_SOURCE_DIR}/cmake/create_test.cmake)

find_package(Kokkos 4.1.00 QUIET)
if(Kokkos_FOUND)
    add_subdirectory(kokkos)
endif()

add_subdirectory(config)

if(GINKGO_BUILD_CUDA)
    add_subdirectory(cuda)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/create_test.cmake)

if (GINKGO_EXTENSION_KOKKOS)
    add_subdirectory(kokkos)
endif ()

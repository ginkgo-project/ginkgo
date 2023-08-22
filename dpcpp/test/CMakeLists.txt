include(${PROJECT_SOURCE_DIR}/cmake/create_test.cmake)
set(GINKGO_COMPILING_SYCL_TEST ON)

add_subdirectory(base)
add_subdirectory(components)
add_subdirectory(matrix)
add_subdirectory(preconditioner)

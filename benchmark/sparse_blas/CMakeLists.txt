ginkgo_add_typed_benchmark_executables(sparse_blas "NO" sparse_blas.cpp operations.cpp)
include(ExternalProject)
find_package(Git REQUIRED)
ExternalProject_Add(SuiteSparse
    GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse
    GIT_TAG 76f266cfd35812b12a8683b8e4ce90a80d718fbe
    PATCH_COMMAND ${GIT_EXECUTABLE} checkout . && ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_SOURCE_DIR}/cholmod_symbolic.patch"
    CONFIGURE_COMMAND true
    BUILD_IN_SOURCE ON
    BUILD_COMMAND make local
    INSTALL_COMMAND make install
    )
ExternalProject_Get_property(SuiteSparse SOURCE_DIR)
add_dependencies(sparse_blas_single SuiteSparse)
add_dependencies(sparse_blas SuiteSparse)
target_link_libraries(sparse_blas_single ${SOURCE_DIR}/lib/libcholmod.so)
target_include_directories(sparse_blas_single PRIVATE "${SOURCE_DIR}/include")
target_link_libraries(sparse_blas ${SOURCE_DIR}/lib/libcholmod.so)
target_include_directories(sparse_blas PRIVATE "${SOURCE_DIR}/include")

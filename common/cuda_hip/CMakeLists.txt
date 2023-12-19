set(CUDA_HIP_SOURCES
    base/device_matrix_data_kernels.cpp
    components/prefix_sum_kernels.cpp
    distributed/matrix_kernels.cpp
    distributed/partition_helpers_kernels.cpp
    distributed/partition_kernels.cpp
    distributed/vector_kernels.cpp
    factorization/factorization_kernels.cpp
    factorization/lu_kernels.cpp
    factorization/par_ic_kernels.cpp
    matrix/coo_kernels.cpp
    matrix/diagonal_kernels.cpp
    matrix/sellp_kernels.cpp
    multigrid/pgm_kernels.cpp
    preconditioner/isai_kernels.cpp
    reorder/rcm_kernels.cpp
    solver/multigrid_kernels.cpp
    )
list(TRANSFORM CUDA_HIP_SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(GKO_CUDA_HIP_COMMON_SOURCES ${CUDA_HIP_SOURCES} PARENT_SCOPE)

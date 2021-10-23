set(UNIFIED_SOURCES
    components/precision_conversion.cpp
    components/reduce_array.cpp
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/diagonal_kernels.cpp
    preconditioner/jacobi_kernels.cpp
    solver/bicg_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/cg_kernels.cpp
    solver/cgs_kernels.cpp
    solver/fcg_kernels.cpp
    solver/ir_kernels.cpp
    )
list(TRANSFORM UNIFIED_SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/unified/)
set(GKO_UNIFIED_COMMON_SOURCES ${UNIFIED_SOURCES} PARENT_SCOPE)

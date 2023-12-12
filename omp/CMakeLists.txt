add_library(ginkgo_omp $<TARGET_OBJECTS:ginkgo_omp_device> "")
# we don't split up the dense kernels into distinct compilations
list(APPEND GKO_UNIFIED_COMMON_SOURCES ${PROJECT_SOURCE_DIR}/common/unified/matrix/dense_kernels.instantiate.cpp)
target_sources(ginkgo_omp
    PRIVATE
    base/batch_multi_vector_kernels.cpp
    base/device_matrix_data_kernels.cpp
    base/executor.cpp
    base/index_set_kernels.cpp
    base/scoped_device_id.cpp
    base/version.cpp
    components/prefix_sum_kernels.cpp
    distributed/matrix_kernels.cpp
    distributed/partition_helpers_kernels.cpp
    distributed/partition_kernels.cpp
    distributed/vector_kernels.cpp
    factorization/cholesky_kernels.cpp
    factorization/factorization_kernels.cpp
    factorization/ic_kernels.cpp
    factorization/ilu_kernels.cpp
    factorization/lu_kernels.cpp
    factorization/par_ic_kernels.cpp
    factorization/par_ict_kernels.cpp
    factorization/par_ilu_kernels.cpp
    factorization/par_ilut_kernels.cpp
    matrix/batch_csr_kernels.cpp
    matrix/batch_dense_kernels.cpp
    matrix/batch_ell_kernels.cpp
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/diagonal_kernels.cpp
    matrix/ell_kernels.cpp
    matrix/fbcsr_kernels.cpp
    matrix/fft_kernels.cpp
    matrix/sellp_kernels.cpp
    matrix/sparsity_csr_kernels.cpp
    multigrid/pgm_kernels.cpp
    preconditioner/isai_kernels.cpp
    preconditioner/jacobi_kernels.cpp
    reorder/rcm_kernels.cpp
    solver/batch_bicgstab_kernels.cpp
    solver/cb_gmres_kernels.cpp
    solver/idr_kernels.cpp
    solver/lower_trs_kernels.cpp
    solver/multigrid_kernels.cpp
    solver/upper_trs_kernels.cpp
    stop/criterion_kernels.cpp
    stop/residual_norm_kernels.cpp
    ${GKO_UNIFIED_COMMON_SOURCES}
    )

ginkgo_compile_features(ginkgo_omp)
target_compile_definitions(ginkgo_omp PRIVATE GKO_COMPILING_OMP)

# TODO FIXME: Currently nvhpc 22.7+ optimizations break the omp jacobi's custom
# precision implementation (mantissa segmentation)
#
# We compile with -O1 to disable the offending optimization. Note that all OMP
# jacobi kernels will be slow on NVHPC as a result.
if (CMAKE_CXX_COMPILER_ID MATCHES "PGI|NVHPC")
    set_source_files_properties(preconditioner/jacobi_kernels.cpp PROPERTIES COMPILE_FLAGS "-O1")
endif()

# The dependency to OpenMP's potential pthread dependency needs to be public
# because well, it's a long story:
# std::shared_ptr decides whether we need to increment/decrement the reference
# counters atomically by calling the pthread function __gthread_active_p, i.e.
# checking whether there are multiple active threads. Linking against OpenMP
# also links against pthread on Linux. Due to issues with static initialization
# order for pthread_key_create, it may happen that ginkgo_omp assumes that a
# piece of code is being executed sequentially, causing race conditions in the
# reference counters.
target_link_libraries(ginkgo_omp PUBLIC Threads::Threads)
target_link_libraries(ginkgo_omp PRIVATE "${OpenMP_CXX_LIBRARIES}")
target_include_directories(ginkgo_omp PRIVATE "${OpenMP_CXX_INCLUDE_DIRS}")
# We first separate the arguments, otherwise, the target_compile_options adds it as a string
# and the compiler is unhappy with the quotation marks.
separate_arguments(OpenMP_SEP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
target_compile_options(ginkgo_omp PRIVATE "${OpenMP_SEP_FLAGS}")
target_compile_options(ginkgo_omp PRIVATE "${GINKGO_COMPILER_FLAGS}")

# Need to link against ginkgo_cuda for the `raw_copy_to(CudaExecutor ...)` method
target_link_libraries(ginkgo_omp PRIVATE ginkgo_cuda)
# Need to link against ginkgo_hip for the `raw_copy_to(HipExecutor ...)` method
target_link_libraries(ginkgo_omp PRIVATE ginkgo_hip)
# Need to link against ginkgo_dpcpp for the `raw_copy_to(DpcppExecutor ...)` method
target_link_libraries(ginkgo_omp PRIVATE ginkgo_dpcpp)
target_link_libraries(ginkgo_omp PUBLIC ginkgo_device)

ginkgo_default_includes(ginkgo_omp)
ginkgo_install_library(ginkgo_omp)

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_omp GKO_COMPILING_OMP)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

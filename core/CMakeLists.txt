add_subdirectory(device_hooks)  # placeholders for disabled modules

add_library(ginkgo "")
target_sources(ginkgo
    PRIVATE
    base/array.cpp
    base/combination.cpp
    base/composition.cpp
    base/dense_cache.cpp
    base/device_matrix_data.cpp
    base/executor.cpp
    base/index_set.cpp
    base/mpi.cpp
    base/mtx_io.cpp
    base/perturbation.cpp
    base/version.cpp
    distributed/partition.cpp
    factorization/elimination_forest.cpp
    factorization/factorization.cpp
    factorization/ic.cpp
    factorization/ilu.cpp
    factorization/lu.cpp
    factorization/par_ic.cpp
    factorization/par_ict.cpp
    factorization/par_ilu.cpp
    factorization/par_ilut.cpp
    factorization/symbolic.cpp
	log/batch_convergence.cpp
    log/convergence.cpp
    log/logger.cpp
    log/performance_hint.cpp
    log/record.cpp
    log/stream.cpp
    matrix/batch_csr.cpp
    matrix/batch_dense.cpp
    matrix/batch_diagonal.cpp
    matrix/batch_ell.cpp
    matrix/coo.cpp
    matrix/csr.cpp
    matrix/dense.cpp
    matrix/diagonal.cpp
    matrix/ell.cpp
    matrix/fbcsr.cpp
    matrix/fft.cpp
    matrix/hybrid.cpp
    matrix/identity.cpp
    matrix/permutation.cpp
    matrix/row_gatherer.cpp
    matrix/sellp.cpp
    matrix/sparsity_csr.cpp
    matrix/row_gatherer.cpp
    multigrid/pgm.cpp
    multigrid/fixed_coarsening.cpp
    preconditioner/batch_ilu.cpp
    preconditioner/batch_ilu_isai.cpp
    preconditioner/batch_isai.cpp
    preconditioner/isai.cpp
    preconditioner/jacobi.cpp
    reorder/rcm.cpp
    reorder/scaled_reordered.cpp
    solver/batch_bicgstab.cpp
    solver/batch_cg.cpp
    solver/batch_direct.cpp
    solver/batch_gmres.cpp
    solver/batch_idr.cpp
    solver/batch_richardson.cpp
    solver/bicg.cpp
    solver/bicgstab.cpp
    solver/cb_gmres.cpp
    solver/cg.cpp
    solver/cgs.cpp
    solver/direct.cpp
    solver/fcg.cpp
    solver/gmres.cpp
    solver/idr.cpp
    solver/ir.cpp
    solver/lower_trs.cpp
    solver/multigrid.cpp
    solver/upper_trs.cpp
    stop/combined.cpp
    stop/criterion.cpp
    stop/iteration.cpp
    stop/residual_norm.cpp
    stop/time.cpp)

if(GINKGO_HAVE_PAPI_SDE)
    target_sources(ginkgo PRIVATE log/papi.cpp)
endif()

if (GINKGO_BUILD_MPI)
    target_sources(ginkgo
        PRIVATE
        mpi/exception.cpp
        distributed/matrix.cpp
        distributed/vector.cpp)
endif()

ginkgo_compile_features(ginkgo)

target_compile_options(ginkgo PRIVATE "${GINKGO_COMPILER_FLAGS}")
# add a namespace alias so Ginkgo can always be included as Ginkgo::ginkgo
# regardless of whether it is installed or added as a subdirectory
add_library(Ginkgo::ginkgo ALIAS ginkgo)
target_link_libraries(ginkgo
    PUBLIC ginkgo_device ginkgo_omp ginkgo_cuda ginkgo_reference ginkgo_hip ginkgo_dpcpp)
# The PAPI dependency needs to be exposed to the user.
set(GKO_RPATH_ADDITIONS "")
if (GINKGO_HAVE_PAPI_SDE)
    target_link_libraries(ginkgo PUBLIC PAPI::PAPI)
    list(GET PAPI_LIBRARIES 0 PAPI_FIRST_LIB)
    get_filename_component(GKO_PAPI_LIBDIR "${PAPI_FIRST_LIB}" DIRECTORY)
    list(APPEND GKO_RPATH_ADDITIONS "${GKO_PAPI_LIBDIR}")
endif()

if (GINKGO_BUILD_MPI)
    target_link_libraries(ginkgo PUBLIC MPI::MPI_CXX)
endif()

# Since we have a public dependency on HIP, this dependency appears
# here as well
if (GINKGO_BUILD_HIP AND GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
    list(APPEND GKO_RPATH_ADDITIONS "${HIP_PATH}/lib")
endif()

ginkgo_default_includes(ginkgo)
ginkgo_install_library(ginkgo "${GKO_RPATH_ADDITIONS}")

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo "")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

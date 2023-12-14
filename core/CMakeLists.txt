add_subdirectory(device_hooks) # placeholders for disabled modules

add_library(ginkgo "")
target_sources(ginkgo
    PRIVATE
    base/array.cpp
    base/batch_multi_vector.cpp
    base/block_operator.cpp
    base/combination.cpp
    base/composition.cpp
    base/dense_cache.cpp
    base/device_matrix_data.cpp
    base/executor.cpp
    base/index_set.cpp
    base/memory.cpp
    base/mpi.cpp
    base/mtx_io.cpp
    base/perturbation.cpp
    base/timer.cpp
    base/version.cpp
    distributed/partition.cpp
    factorization/cholesky.cpp
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
    log/batch_logger.cpp
    log/convergence.cpp
    log/logger.cpp
    log/performance_hint.cpp
    log/profiler_hook.cpp
    log/profiler_hook_summary.cpp
    log/profiler_hook_summary_writer.cpp
    log/tau.cpp
    log/vtune.cpp
    log/record.cpp
    log/stream.cpp
    matrix/batch_csr.cpp
    matrix/batch_dense.cpp
    matrix/batch_ell.cpp
    matrix/batch_identity.cpp
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
    matrix/scaled_permutation.cpp
    matrix/sellp.cpp
    matrix/sparsity_csr.cpp
    multigrid/pgm.cpp
    multigrid/fixed_coarsening.cpp
    preconditioner/isai.cpp
    preconditioner/jacobi.cpp
    reorder/amd.cpp
    reorder/mc64.cpp
    reorder/rcm.cpp
    reorder/scaled_reordered.cpp
    solver/batch_bicgstab.cpp
    solver/bicg.cpp
    solver/bicgstab.cpp
    solver/cb_gmres.cpp
    solver/cg.cpp
    solver/cgs.cpp
    solver/direct.cpp
    solver/fcg.cpp
    solver/gcr.cpp
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
    stop/time.cpp
    ../third_party/SuiteSparse/wrapper32.cpp
    ../third_party/SuiteSparse/wrapper64.cpp
    )

if(GINKGO_HAVE_PAPI_SDE)
    target_sources(ginkgo PRIVATE log/papi.cpp)
endif()

if(GINKGO_HAVE_METIS)
    target_sources(ginkgo PRIVATE reorder/nested_dissection.cpp)
endif()

if(GINKGO_BUILD_MPI)
    target_sources(ginkgo
        PRIVATE
        mpi/exception.cpp
        distributed/matrix.cpp
        distributed/partition_helpers.cpp
        distributed/vector.cpp
        distributed/preconditioner/schwarz.cpp)
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

if(GINKGO_HAVE_PAPI_SDE)
    target_link_libraries(ginkgo PUBLIC PAPI::PAPI_SDE)
endif()

if(GINKGO_HAVE_TAU)
    target_link_libraries(ginkgo PRIVATE perfstubs)
endif()

if(GINKGO_HAVE_VTUNE)
    target_link_libraries(ginkgo PRIVATE VTune::ITT)
endif()

if(GINKGO_HAVE_METIS)
    target_link_libraries(ginkgo PRIVATE METIS::METIS)
endif()

if(GINKGO_BUILD_MPI)
    target_link_libraries(ginkgo PUBLIC MPI::MPI_CXX)
endif()

# Since we have a public dependency on HIP, this dependency appears
# here as well
if(GINKGO_BUILD_HIP AND GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
    list(APPEND GKO_RPATH_ADDITIONS "${HIP_PATH}/lib")
endif()

ginkgo_default_includes(ginkgo)
ginkgo_install_library(ginkgo "${GKO_RPATH_ADDITIONS}")

if(GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo "")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

if(GINKGO_DEVEL_TOOLS AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BUILD_SHARED_LIBS)
    # Copy pretty-printer next to library
    add_custom_command(TARGET ginkgo POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
        "${Ginkgo_SOURCE_DIR}/dev_tools/scripts/gdb-ginkgo.py"
        "$<TARGET_FILE:ginkgo>-gdb.py"
        COMMENT "Copying pretty-printer to output directory")
endif()

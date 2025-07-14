add_subdirectory(device_hooks) # placeholders for disabled modules

set(config_source
    config/factorization_config.cpp
    config/multigrid_config.cpp
    config/preconditioner_config.cpp
    config/preconditioner_ic_config.cpp
    config/preconditioner_ilu_config.cpp
    config/preconditioner_isai_config.cpp
    config/registry.cpp
    config/solver_config.cpp
)
if(GINKGO_BUILD_MPI)
    list(APPEND config_source config/schwarz_config.cpp)
endif()
# MSVC: LNK1189 issue
# CLANG in MSYS2 (MINGW): too many exported symbols
# We separate the library as a workaround to solve this issue
# To make ginkgo still be the major library, we make the original to ginkgo_core in MSVC/shared
# TODO: should think another way to solve it like dllexport or def file
set(ginkgo_core "ginkgo")
if((MSVC OR MINGW) AND BUILD_SHARED_LIBS)
    set(ginkgo_core "ginkgo_core")
endif()

add_library(${ginkgo_core} "")
target_sources(
    ${ginkgo_core}
    PRIVATE
        ../third_party/SuiteSparse/wrapper32.cpp
        ../third_party/SuiteSparse/wrapper64.cpp
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
        base/segmented_array.cpp
        base/timer.cpp
        base/version.cpp
        components/range_minimum_query.cpp
        config/config.cpp
        config/config_helper.cpp
        config/property_tree.cpp
        config/stop_config.cpp
        config/type_descriptor.cpp
        distributed/index_map.cpp
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
        log/record.cpp
        log/solver_progress.cpp
        log/stream.cpp
        log/tau.cpp
        log/vtune.cpp
        matrix/batch_csr.cpp
        matrix/batch_dense.cpp
        matrix/batch_ell.cpp
        matrix/batch_identity.cpp
        matrix/coo.cpp
        matrix/csr.cpp
        matrix/csr_lookup.cpp
        matrix/dense.cpp
        matrix/diagonal.cpp
        matrix/ell.cpp
        matrix/fbcsr.cpp
        matrix/fft.cpp
        matrix/hybrid.cpp
        matrix/identity.cpp
        matrix/multivector.cpp
        matrix/permutation.cpp
        matrix/row_gatherer.cpp
        matrix/scaled_permutation.cpp
        matrix/sellp.cpp
        matrix/sparsity_csr.cpp
        multigrid/fixed_coarsening.cpp
        multigrid/pgm.cpp
        preconditioner/batch_jacobi.cpp
        preconditioner/gauss_seidel.cpp
        preconditioner/ic.cpp
        preconditioner/ilu.cpp
        preconditioner/isai.cpp
        preconditioner/jacobi.cpp
        preconditioner/sor.cpp
        reorder/amd.cpp
        reorder/mc64.cpp
        reorder/rcm.cpp
        reorder/scaled_reordered.cpp
        solver/batch_bicgstab.cpp
        solver/batch_cg.cpp
        solver/bicg.cpp
        solver/bicgstab.cpp
        solver/cb_gmres.cpp
        solver/cg.cpp
        solver/cgs.cpp
        solver/chebyshev.cpp
        solver/direct.cpp
        solver/fcg.cpp
        solver/gcr.cpp
        solver/gmres.cpp
        solver/idr.cpp
        solver/ir.cpp
        solver/lower_trs.cpp
        solver/minres.cpp
        solver/multigrid.cpp
        solver/pipe_cg.cpp
        solver/upper_trs.cpp
        stop/combined.cpp
        stop/criterion.cpp
        stop/iteration.cpp
        stop/residual_norm.cpp
        stop/time.cpp
)

if(GINKGO_HAVE_PAPI_SDE)
    target_sources(${ginkgo_core} PRIVATE log/papi.cpp)
endif()

if(GINKGO_HAVE_METIS)
    target_sources(${ginkgo_core} PRIVATE reorder/nested_dissection.cpp)
endif()

if(GINKGO_BUILD_MPI)
    target_sources(
        ${ginkgo_core}
        PRIVATE
            distributed/assembly.cpp
            distributed/collective_communicator.cpp
            distributed/dense_communicator.cpp
            distributed/matrix.cpp
            distributed/neighborhood_communicator.cpp
            distributed/partition_helpers.cpp
            distributed/preconditioner/schwarz.cpp
            distributed/row_gatherer.cpp
            distributed/vector.cpp
            distributed/vector_cache.cpp
            mpi/exception.cpp
    )
endif()

# MSVC or CLANG/msys2 with shared: make ginkgo be the major library
if((MSVC OR MINGW) AND BUILD_SHARED_LIBS)
    add_library(ginkgo "")
    target_sources(ginkgo PRIVATE ${config_source})
    ginkgo_compile_features(ginkgo)
    ginkgo_default_includes(ginkgo)
    ginkgo_install_library(ginkgo)
    if(GINKGO_CHECK_CIRCULAR_DEPS)
        ginkgo_check_headers(ginkgo "")
    endif()
else()
    target_sources(${ginkgo_core} PRIVATE ${config_source})
endif()

ginkgo_compile_features(${ginkgo_core})

# add a namespace alias so Ginkgo can always be included as Ginkgo::ginkgo
# regardless of whether it is installed or added as a subdirectory
add_library(Ginkgo::ginkgo ALIAS ginkgo)
if((MSVC OR MINGW) AND BUILD_SHARED_LIBS)
    target_link_libraries(ginkgo PUBLIC ${ginkgo_core})
endif()
target_link_libraries(
    ${ginkgo_core}
    PUBLIC
        ginkgo_device
        ginkgo_omp
        ginkgo_cuda
        ginkgo_reference
        ginkgo_hip
        ginkgo_dpcpp
)
if(GINKGO_HAVE_PAPI_SDE)
    target_link_libraries(${ginkgo_core} PUBLIC PAPI::PAPI_SDE)
endif()

if(GINKGO_HAVE_TAU)
    target_link_libraries(${ginkgo_core} PRIVATE perfstubs)
endif()

if(GINKGO_HAVE_VTUNE)
    target_link_libraries(${ginkgo_core} PRIVATE VTune::ITT)
endif()

if(GINKGO_HAVE_METIS)
    target_link_libraries(${ginkgo_core} PRIVATE METIS::METIS)
endif()

if(GINKGO_BUILD_MPI)
    target_link_libraries(${ginkgo_core} PUBLIC MPI::MPI_CXX)
endif()

ginkgo_default_includes(${ginkgo_core})
ginkgo_install_library(${ginkgo_core})

if(GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(${ginkgo_core} "")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

if(
    GINKGO_DEVEL_TOOLS
    AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
    AND BUILD_SHARED_LIBS
)
    # Copy pretty-printer next to library
    add_custom_command(
        TARGET ${ginkgo_core}
        POST_BUILD
        COMMAND
            "${CMAKE_COMMAND}" -E copy
            "${Ginkgo_SOURCE_DIR}/dev_tools/scripts/gdb-ginkgo.py"
            "$<TARGET_FILE:ginkgo>-gdb.py"
        COMMENT "Copying pretty-printer to output directory"
    )
endif()

add_subdirectory(devices)  # basic device functionalities, always compiled
add_subdirectory(device_hooks)  # placeholders for disabled modules

add_library(ginkgo "")
target_sources(ginkgo
    PRIVATE
        base/combination.cpp
        base/composition.cpp
        base/executor.cpp
        base/mtx_io.cpp
        base/version.cpp
        factorization/par_ilu.cpp
        log/convergence.cpp
        log/logger.cpp
        log/record.cpp
        log/stream.cpp
        matrix/coo.cpp
        matrix/csr.cpp
        matrix/dense.cpp
        matrix/ell.cpp
        matrix/hybrid.cpp
        matrix/identity.cpp
        matrix/sellp.cpp
        preconditioner/jacobi.cpp
        solver/bicgstab.cpp
        solver/cg.cpp
        solver/cgs.cpp
        solver/fcg.cpp
        solver/gmres.cpp
        solver/ir.cpp
        stop/combined.cpp
        stop/criterion.cpp
        stop/iteration.cpp
        stop/residual_norm_reduction.cpp
        stop/time.cpp)

if(GINKGO_HAVE_PAPI_SDE)
    target_sources(ginkgo PRIVATE log/papi.cpp)
endif()

ginkgo_compile_features(ginkgo)

target_compile_options(ginkgo PRIVATE "${GINKGO_COMPILER_FLAGS}")
# add a namespace alias so Ginkgo can always be included as Ginkgo::ginkgo
# regardless of whether it is installed or added as a subdirectory
add_library(Ginkgo::ginkgo ALIAS ginkgo)
target_link_libraries(ginkgo
    PUBLIC ginkgo_omp ginkgo_cuda ginkgo_reference)
if (GINKGO_HAVE_PAPI_SDE)
    target_link_libraries(ginkgo PRIVATE PAPI::PAPI)
endif()
ginkgo_default_includes(ginkgo)
ginkgo_install_library(ginkgo core)

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()


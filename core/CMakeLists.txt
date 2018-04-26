set(SOURCES
    base/executor.cpp
    base/mtx_reader.cpp
    log/stream.cpp
    log/logger.cpp
    log/record.cpp
    matrix/coo.cpp
    matrix/csr.cpp
    matrix/dense.cpp
    matrix/ell.cpp
    matrix/hybrid.cpp
    matrix/identity.cpp
    matrix/sellp.cpp
    preconditioner/block_jacobi.cpp
    solver/bicgstab.cpp
    solver/cg.cpp
    solver/cgs.cpp
    solver/fcg.cpp)


add_subdirectory(devices)  # basic device functionalities, always compiled
add_subdirectory(device_hooks)  # placeholders for disabled modules

if(BUILD_SHARED_LIBS)
    set_property(TARGET mmio PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
add_library(ginkgo ${SOURCES})
# add a namespace alias so Ginkgo can always be included as Ginkgo::ginkgo
# regardless of whether it is installed or added as a subdirectory
add_library(Ginkgo::ginkgo ALIAS ginkgo)
target_link_libraries(ginkgo
    PUBLIC ginkgo_omp ginkgo_gpu ginkgo_reference
    PRIVATE mmio)
ginkgo_default_includes(ginkgo)
ginkgo_install_library(ginkgo core)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()


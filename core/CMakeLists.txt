set(SOURCES
    base/executor.cpp
    base/mtx_reader.cpp
    matrix/coo.cpp
    matrix/csr.cpp
    matrix/dense.cpp
    matrix/ell.cpp
    matrix/hybrid.cpp
    matrix/identity.cpp
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
add_library(Ginkgo::ginkgo ALIAS ginkgo)
target_link_libraries(ginkgo PRIVATE mmio)
if(BUILD_CPU)
    target_link_libraries(ginkgo PUBLIC ginkgo_cpu)
endif()
if(BUILD_GPU)
    target_link_libraries(ginkgo PUBLIC ginkgo_gpu)
endif()
if(BUILD_REFERENCE)
    target_link_libraries(ginkgo PUBLIC ginkgo_reference)
endif()
ginkgo_default_includes(ginkgo)
ginkgo_install_library(ginkgo core)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()


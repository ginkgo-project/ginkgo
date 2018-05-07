set(SOURCES
    base/executor.cpp
    base/mtx_reader.cpp
    log/ostream.cpp
    log/logger.cpp # WTF C++
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
target_include_directories(ginkgo PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(ginkgo
    PUBLIC ginkgo_cpu ginkgo_gpu ginkgo_reference
    PRIVATE mmio)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()


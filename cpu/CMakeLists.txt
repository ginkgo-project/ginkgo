set(SOURCES
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/fcg_kernels.cpp
    solver/cg_kernels.cpp)

add_library(ginkgo_cpu SHARED
    $<TARGET_OBJECTS:ginkgo_cpu_device>
    ${SOURCES})
target_include_directories(ginkgo_cpu PUBLIC ${PROJECT_SOURCE_DIR})

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

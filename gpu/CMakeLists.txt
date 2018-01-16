set(SOURCES
    base/exception.cpp
    base/executor.cpp
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/cg_kernels.cpp)

set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_library(CUDA_RUNTIME_LIBS cudart
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS cublas
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_library(ginkgo_gpu SHARED
    $<TARGET_OBJECTS:ginkgo_gpu_device>
    ${SOURCES})
target_include_directories(ginkgo_gpu PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(ginkgo_gpu
    SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(ginkgo_gpu PRIVATE ${CUDA_RUNTIME_LIBS} ${CUBLAS})

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

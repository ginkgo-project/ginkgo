if (GINKGO_BUILD_CUDA AND GINKGO_BUILD_OMP)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_executable(example-7
        example-7.cpp
        stencil_kernel.cu)
    target_link_libraries(example-7 ginkgo)
    target_include_directories(example-7 PRIVATE
        ${PROJECT_SOURCE_DIR})
endif()

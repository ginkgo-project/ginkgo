if (BUILD_CUDA AND BUILD_OMP)
    enable_language(CUDA)
    add_executable(custom_matrix_format
        custom_matrix_format.cpp
        stencil_kernel.cu)
    target_link_libraries(custom_matrix_format ginkgo)
    target_include_directories(custom_matrix_format PRIVATE
        ${PROJECT_SOURCE_DIR})
endif()

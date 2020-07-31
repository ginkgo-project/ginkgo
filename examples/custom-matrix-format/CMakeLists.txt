if (GINKGO_BUILD_CUDA AND GINKGO_BUILD_OMP)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_executable(custom-matrix-format
        custom-matrix-format.cpp
        stencil_kernel.cu)
    target_link_libraries(custom-matrix-format ginkgo)
    target_include_directories(custom-matrix-format PRIVATE
        ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    # workaround for clang-cuda/g++ interaction
    set_target_properties(custom-matrix-format PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

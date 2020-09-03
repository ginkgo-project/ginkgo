if (GINKGO_BUILD_CUDA AND GINKGO_BUILD_OMP)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(run_target "custom-matrix-format")
    add_executable(${run_target}
        ${run_target}.cpp
        stencil_kernel.cu)
    target_link_libraries(${run_target} ginkgo)
    target_include_directories(${run_target} PRIVATE
        ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    # workaround for clang-cuda/g++ interaction
    set_target_properties(${run_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if (GINKGO_BUILD_CUDA AND GINKGO_BUILD_OMP)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(target_name "custom-matrix-format")
    add_executable(${target_name}
        ${target_name}.cpp
        stencil_kernel.cu)
    target_link_libraries(${target_name} Ginkgo::ginkgo)
    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    # workaround for clang-cuda/g++ interaction
    set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

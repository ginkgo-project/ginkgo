if (BUILD_CUDA)
    add_executable(simple_spmv simple_spmv.cpp)
    target_link_libraries(simple_spmv ginkgo ${CUDA_RUNTIME_LIBS}
        ${CUBLAS} ${CUSPARSE})
    target_include_directories(simple_spmv PRIVATE ${PROJECT_SOURCE_DIR})
    if(cuda_version GREATER_EQUAL "92")
        add_definitions(-DALLOWMP)
    endif()
endif()


cuda_add_executable(gpu_kernel gpu_kernel.cu) #will not work because the file
                                              #does not contain main, but just
                                              #for commit

target_link_libraries(gpu_kernel ginkgo_gpu ${CUDA_LIBRARIES})

create_test(gpu_executor)

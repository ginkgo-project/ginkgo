if(GINKGO_BUILD_CUDA)
    find_package(cudss 0.7.1 CONFIG)
endif()

add_subdirectory(cuda)
# if(GINKGO_BUILD_TESTS)
#     add_subdirectory(test)
# endif()

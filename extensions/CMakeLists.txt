add_subdirectory(cuda)
if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

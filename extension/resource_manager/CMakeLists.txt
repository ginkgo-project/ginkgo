if(GINKGO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

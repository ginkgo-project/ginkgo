if(NOT GINKGO_BUILD_CUDA)
    return()
endif()

if(NOT cudss_FOUND)
    return()
endif()

ginkgo_create_test(solver/cudss ADDITIONAL_LIBRARIES ginkgo_cudss)

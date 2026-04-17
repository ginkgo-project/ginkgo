if(NOT GINKGO_BUILD_CUDA)
    return()
endif()

find_package(cuDSS QUIET)
if(NOT cuDSS_FOUND)
    return()
endif()

ginkgo_create_test(solver/cudss)

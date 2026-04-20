if(NOT GINKGO_BUILD_CUDA)
    return()
endif()

find_package(cudss 0.7.1 CONFIG QUIET)
if(NOT cudss_FOUND)
    return()
endif()

ginkgo_create_test(solver/cudss)

if(MSVC AND BUILD_SHARED_LIBS)
    # Force using shared runtime library when MSVC builds shared libraries
    ginkgo_load_git_package(gtest_external
        "https://github.com/google/googletest.git"
        "2fe3bd994b3189899d93f1d5a881e725e046fdc2"
        # Work around the linking errors when compiling gtest with CUDA
        "-Dgtest_disable_pthreads=ON" "-Dgtest_force_shared_crt=TRUE")
else()
    ginkgo_load_git_package(gtest_external
        "https://github.com/google/googletest.git"
        "2fe3bd994b3189899d93f1d5a881e725e046fdc2"
        # Work around the linking errors when compiling gtest with CUDA
        "-Dgtest_disable_pthreads=ON")
endif()
ginkgo_add_external_target(GTest::GTest gtest src/googletest/include build/googlemock/gtest
    STATIC "d" gtest_external FALSE)
ginkgo_add_external_target(GTest::Main gtest_main src/googletest/include build/googlemock/gtest
    STATIC "d" gtest_external FALSE)

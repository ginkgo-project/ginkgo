message(STATUS "Configuring and building gtest")
if(MSVC)
    # Force using shared runtime library when MSVC builds shared libraries
    ginkgo_load_git_package(gtest_external
        "https://github.com/google/googletest.git"
        "6a7ed316a5cdc07b6d26362c90770787513822d4"
        # Work around the linking errors when compiling gtest with CUDA
        "-Dgtest_disable_pthreads=ON" "-Dgtest_force_shared_crt=${BUILD_SHARED_LIBS}")
else()
    ginkgo_load_git_package(gtest_external
        "https://github.com/google/googletest.git"
        "6a7ed316a5cdc07b6d26362c90770787513822d4"
        # Work around the linking errors when compiling gtest with CUDA
        "-Dgtest_disable_pthreads=ON"
        "-DCMAKE_CXX_FLAGS=-fPIC")
endif()

ginkgo_add_external_target(GTest::GTest gtest src/googletest/include build/lib
    STATIC "d" gtest_external FALSE)
ginkgo_add_external_target(GTest::Main gtest_main src/googletest/include build/lib
    STATIC "d" gtest_external FALSE)

ginkgo_load_git_package(gtest_external
    "https://github.com/google/googletest.git"
    "d5266326752f0a1dadbd310932d8f4fd8c3c5e7d"
    # Work around the linking errors when compiling gtest with CUDA
    "-Dgtest_disable_pthreads=ON")
ginkgo_add_external_target(gtest src/googletest/include build/googlemock/gtest
    STATIC "d" gtest_external FALSE)
ginkgo_add_external_target(gtest_main src/googletest/include build/googlemock/gtest
    STATIC "d" gtest_external FALSE)
add_library(GTest::GTest ALIAS gtest)
add_library(GTest::Main ALIAS gtest_main)

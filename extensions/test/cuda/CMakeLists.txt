if(NOT GINKGO_BUILD_CUDA)
    return()
endif()

find_package(cuDSS QUIET)
if(NOT cuDSS_FOUND)
    return()
endif()

add_executable(test_ext_cudss solver/cudss.cpp)

target_link_libraries(test_ext_cudss PRIVATE ginkgo ginkgo_cudss GTest::Main)

target_include_directories(
    test_ext_cudss
    PRIVATE ${Ginkgo_SOURCE_DIR} ${Ginkgo_BINARY_DIR}
)

gtest_add_tests(TARGET test_ext_cudss)

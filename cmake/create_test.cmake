function(ginkgo_create_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
         ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
            "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::GTest GTest::Main ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_test)

function(ginkgo_create_cuda_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
         ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cu)
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
            "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::GTest GTest::Main ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_cuda_test)

function(ginkgo_create_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
         ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo gtest_main)
    # Needed because it causes undefined reference errors on some systems.
    # To prevent that, the library files are added to the linker again.
    target_link_libraries(${TEST_TARGET_NAME} 
        PRIVATE $<TARGET_LINKER_FILE:ginkgo_cuda>
                $<TARGET_LINKER_FILE:ginkgo_reference>
                $<TARGET_LINKER_FILE:ginkgo_omp>
                $<TARGET_LINKER_FILE:ginkgo>)
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_test)

function(ginkgo_create_cuda_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
         ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cu)
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo gtest_main)
    # Needed because it causes undefined reference errors on some systems.
    # To prevent that, the library files are added to the linker again.
    target_link_libraries(${TEST_TARGET_NAME} 
        PRIVATE $<TARGET_LINKER_FILE:ginkgo_cuda>
                $<TARGET_LINKER_FILE:ginkgo>)
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_cuda_test)

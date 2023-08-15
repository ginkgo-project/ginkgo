function(gkoext_pt_create_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
    target_compile_options("${TEST_TARGET_NAME}" PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "${CMAKE_SOURCE_DIR}/extension/property_tree/include"
        "${CMAKE_SOURCE_DIR}/extension/property_tree"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(gkoext_pt_create_test)


gkoext_pt_create_test(json_parser rapidjson)
gkoext_pt_create_test(string_parser)
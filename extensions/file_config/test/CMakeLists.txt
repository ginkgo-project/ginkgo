function(gkoext_file_config_create_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
    target_compile_options("${TEST_TARGET_NAME}" PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        "${CMAKE_SOURCE_DIR}/extensions/file_config/include"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE Ginkgo::ginkgo file_config GTest::Main GTest::GTest nlohmann_json::nlohmann_json ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(gkoext_file_config_create_test)

add_subdirectory(base)
add_subdirectory(custom)
add_subdirectory(executor)
add_subdirectory(log)
add_subdirectory(matrix)
add_subdirectory(preconditioner)
add_subdirectory(stop)

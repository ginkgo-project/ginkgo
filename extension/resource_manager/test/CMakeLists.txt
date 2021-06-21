function(gkoext_manager_create_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
    target_compile_options("${TEST_TARGET_NAME}" PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        "${CMAKE_SOURCE_DIR}/extension/resource_manager/include"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_COMPILING_DPCPP_TEST AND GINKGO_DPCPP_SINGLE_MODE)
        target_compile_definitions("${TEST_TARGET_NAME}" PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
    endif()
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest rapidjson ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(gkoext_manager_create_test)

add_subdirectory(base)
add_subdirectory(executor)
add_subdirectory(matrix)
add_subdirectory(preconditioner)
add_subdirectory(solver)
add_subdirectory(stop)

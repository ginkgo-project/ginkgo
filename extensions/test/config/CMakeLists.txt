ginkgo_create_test(json_config ADDITIONAL_LIBRARIES nlohmann_json::nlohmann_json)
# set the working directory to the current binary folder for file test
file(RELATIVE_PATH REL_BINARY_DIR ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties("${REL_BINARY_DIR}/json_config" PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
configure_file(test.json test.json COPYONLY)

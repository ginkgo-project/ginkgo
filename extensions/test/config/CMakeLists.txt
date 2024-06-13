ginkgo_create_test(json_config ADDITIONAL_LIBRARIES nlohmann_json::nlohmann_json)

# prepare the testing file and generate location
configure_file("${Ginkgo_SOURCE_DIR}/extensions/test/config/file_location.hpp.in"
    "${Ginkgo_BINARY_DIR}/extensions/test/config/file_location.hpp" @ONLY)
configure_file(test.json "${Ginkgo_BINARY_DIR}/extensions/test/config/test.json")

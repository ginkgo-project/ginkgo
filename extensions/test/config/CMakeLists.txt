ginkgo_create_test(json_config ADDITIONAL_LIBRARIES nlohmann_json::nlohmann_json)
configure_file("test.json" test.json COPYONLY)

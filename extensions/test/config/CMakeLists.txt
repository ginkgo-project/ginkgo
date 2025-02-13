find_package(yaml-cpp 0.8.0 QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "Fetching external yaml-cpp")
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0
    )
    FetchContent_MakeAvailable(yaml-cpp)
endif()
ginkgo_create_test(json_config ADDITIONAL_LIBRARIES nlohmann_json::nlohmann_json)
ginkgo_create_test(yaml_config ADDITIONAL_LIBRARIES yaml-cpp::yaml-cpp)

# prepare the testing file and generate location
configure_file(
    "${Ginkgo_SOURCE_DIR}/extensions/test/config/file_location.hpp.in"
    "${Ginkgo_BINARY_DIR}/extensions/test/config/file_location.hpp"
    @ONLY
)
configure_file(
    test.json
    "${Ginkgo_BINARY_DIR}/extensions/test/config/test.json"
    COPYONLY
)
configure_file(
    test.yaml
    "${Ginkgo_BINARY_DIR}/extensions/test/config/test.yaml"
    COPYONLY
)
configure_file(
    alias.yaml
    "${Ginkgo_BINARY_DIR}/extensions/test/config/alias.yaml"
    COPYONLY
)
configure_file(
    nested_alias.yaml
    "${Ginkgo_BINARY_DIR}/extensions/test/config/nested_alias.yaml"
    COPYONLY
)

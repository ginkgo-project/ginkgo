find_package(yaml-cpp 0.8.0 QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "Fetching external yaml-cpp")
    include(FetchContent)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG bbf8bdb087bb3f3621ca0a5ace06047805f4e9f3
    )

    # Turn off additional tool in yaml-cpp
    # We change the standard of yaml-cpp. Without this, we will need to change the standard of tool targets additionally.
    set(YAML_CPP_BUILD_TOOLS OFF CACHE INTERNAL "")
    # Turn off YAML formatting, since this clashes with our formatting target
    set(YAML_CPP_FORMAT_SOURCE OFF CACHE INTERNAL "")

    FetchContent_MakeAvailable(yaml-cpp)
    # make sure the tests DLLs are placed in the working path for CTest
    set_target_properties(
        yaml-cpp
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}"
            ARCHIVE_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}"
            LIBRARY_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}"
    )

    # The default standard c++11, which introduces an issue on dpcpp compiler.
    # We set the default to c++17 as Ginkgo
    set_target_properties(yaml-cpp PROPERTIES CXX_STANDARD 17)
endif()
ginkgo_create_test(
    json_config
    ADDITIONAL_LIBRARIES
    nlohmann_json::nlohmann_json
)
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

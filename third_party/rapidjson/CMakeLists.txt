ginkgo_load_git_package(rapidjson_external
    "https://github.com/Tencent/rapidjson.git"
    "73063f5002612c6bf64fe24f851cd5cc0d83eef9"
    "-DRAPIDJSON_BUILD_DOC=OFF" "-DRAPIDJSON_BUILD_EXAMPLES=OFF"
    "-DRAPIDJSON_BUILD_TESTS=OFF" "-DRAPIDJSON_BUILD_THIRDPARTY_GTEST=OFF")
# Remove directory with questionable licensing
file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/src/bin/jsonchecker)
ginkgo_add_external_target(rapidjson src/include "" STATIC "" rapidjson_external TRUE)

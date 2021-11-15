message(STATUS "Fetching external RapidJSON")
include(FetchContent)
FetchContent_Declare(
    rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        27c3a8dc0e2c9218fe94986d249a12b5ed838f1d
)
FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
    FetchContent_Populate(rapidjson)
endif()

set(RapidJSON_INCLUDE_DIR "${rapidjson_SOURCE_DIR}/include/")
add_library(rapidjson INTERFACE)
target_include_directories(rapidjson INTERFACE
    $<BUILD_INTERFACE:${RapidJSON_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

if (GINKGO_BUILD_EXTENSION)
    ginkgo_install_header_only_library(rapidjson RapidJSON)
    file(READ "${RapidJSON_INCLUDE_DIR}/rapidjson/rapidjson.h"
        HEADER_CONTENTS LIMIT 16384)
    string(REGEX REPLACE ".*#define RAPIDJSON_MAJOR_VERSION ([0-9]+).*" "\\1"
        RAPIDJSON_MAJOR "${HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define RAPIDJSON_MINOR_VERSION ([0-9]+).*" "\\1"
        RAPIDJSON_MINOR "${HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define RAPIDJSON_PATCH_VERSION ([0-9]+).*" "\\1"
        RAPIDJSON_PATCH "${HEADER_CONTENTS}")
    set(RapidJSON_VERSION "${RAPIDJSON_MAJOR}.${RAPIDJSON_MINOR}.${RAPIDJSON_PATCH}")
    write_basic_package_version_file(
        "${Ginkgo_BINARY_DIR}/RapidJSONConfigVersion.cmake"
        VERSION "${RapidJSON_VERSION}"
        COMPATIBILITY SameMajorVersion
        )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/RapidJSONConfig.cmake.in"
        "${Ginkgo_BINARY_DIR}/RapidJSONConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RapidJSON"
        )
    install(FILES
        "${Ginkgo_BINARY_DIR}/RapidJSONConfig.cmake"
        "${Ginkgo_BINARY_DIR}/RapidJSONConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RapidJSON"
        )
    export(EXPORT RapidJSON
        FILE "${Ginkgo_BINARY_DIR}/RapidJSONTargets.cmake"
        )
    install(EXPORT RapidJSON
        FILE RapidJSONTargets.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RapidJSON"
        )
endif()

message(STATUS "Fetching external RapidJSON")
include(FetchContent)
FetchContent_Declare(
    rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        48fbd8cd202ca54031fe799db2ad44ffa8e77c13
)
FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
    FetchContent_Populate(rapidjson)
endif()
set(RapidJSON_INCLUDE_DIR "${rapidjson_SOURCE_DIR}/include")
add_library(rapidjson INTERFACE)
set_target_properties(rapidjson PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${RapidJSON_INCLUDE_DIR}")

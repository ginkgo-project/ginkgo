if(NOT RapidJSON_FOUND)
    message(STATUS "Fetching external RapidJSON")
    include(FetchContent)
    FetchContent_Declare(
        rapidjson
        GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
        GIT_TAG        88bd956d66d348f478bceebfdadb8e26c6844695
    )
    FetchContent_GetProperties(rapidjson)
    if(NOT rapidjson_POPULATED)
        FetchContent_Populate(rapidjson)
    endif()
    set(RapidJSON_INCLUDE_DIR "${rapidjson_SOURCE_DIR}/include")
endif()
# some installed versions of RapidJSON already export a target
if (NOT TARGET rapidjson)
    add_library(rapidjson INTERFACE)
    set_target_properties(rapidjson PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${RapidJSON_INCLUDE_DIR}")
endif()

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

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
set(RapidJSON_INCLUDE_DIR "${rapidjson_SOURCE_DIR}/include")
add_library(rapidjson INTERFACE)
set_target_properties(rapidjson PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${RapidJSON_INCLUDE_DIR}")

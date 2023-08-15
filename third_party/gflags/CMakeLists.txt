# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

message(STATUS "Fetching external GFlags")
include(FetchContent)
FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG        a738fdf9338412f83ab3f26f31ac11ed3f3ec4bd
)
# need to set the variables in CACHE due to CMP0077
set(GFLAGS_BUILD_TESTING OFF CACHE INTERNAL "")
set(GFLAGS_BUILD_PACKAGING OFF CACHE INTERNAL "")
FetchContent_GetProperties(gflags)
if(NOT gflags_POPULATED)
    FetchContent_Populate(gflags)
    add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

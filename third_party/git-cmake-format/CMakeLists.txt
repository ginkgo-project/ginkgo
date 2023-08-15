# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

message(STATUS "Fetching git-cmake-format")
include(FetchContent)
FetchContent_Declare(
        git_cmake_format
        GIT_REPOSITORY https://github.com/ginkgo-project/git-cmake-format.git
        GIT_TAG 26465c1669f3f64bf6709170fb98dcf63561b29e
)
FetchContent_GetProperties(git_cmake_format)
if(NOT git_cmake_format_POPULATED)
    FetchContent_Populate(git_cmake_format)

    set(GCF_FORCE_OVERWRITE ON CACHE INTERNAL "If true, always overwrite pre-commit hook and script")
    set(GCF_GIT_TOP_LEVEL_DIR ${Ginkgo_SOURCE_DIR})
    add_subdirectory(${git_cmake_format_SOURCE_DIR} ${git_cmake_format_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

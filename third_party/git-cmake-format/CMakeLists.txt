message(STATUS "Fetching git-cmake-format")
include(FetchContent)
FetchContent_Declare(
        git_cmake_format
        GIT_REPOSITORY https://github.com/ginkgo-project/git-cmake-format.git
        GIT_TAG customize-toplevel-dir
)
FetchContent_GetProperties(git_cmake_format)
if(NOT git_cmake_format_POPULATED)
    FetchContent_Populate(git_cmake_format)

    set(GCF_FORCE_OVERWRITE ON CACHE INTERNAL "If true, always overwrite pre-commit hook and script")
    set(GCF_GIT_TOP_LEVEL_DIR ${Ginkgo_SOURCE_DIR})
    add_subdirectory(${git_cmake_format_SOURCE_DIR} ${git_cmake_format_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

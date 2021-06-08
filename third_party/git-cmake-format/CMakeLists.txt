message(STATUS "Fetching git-cmake-format")
include(FetchContent)
FetchContent_Declare(
    git_cmake_format
    GIT_REPOSITORY https://github.com/ginkgo-project/git-cmake-format.git
    GIT_TAG        be9554a5d71030dcc23f1eb355c41ab9d876e776
)
FetchContent_GetProperties(git_cmake_format)
if(NOT git_cmake_format_POPULATED)
    FetchContent_Populate(git_cmake_format)
    add_subdirectory(${git_cmake_format_SOURCE_DIR} ${git_cmake_format_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

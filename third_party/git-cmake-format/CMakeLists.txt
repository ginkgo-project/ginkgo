message(STATUS "Configuring and building git-cmake-format")
ginkgo_load_git_package(git-cmake-format
    "https://github.com/ginkgo-project/git-cmake-format.git"
    "be9554a5d71030dcc23f1eb355c41ab9d876e776"
    "-DGCF_CLANGFORMAT_MINIMAL_VERSION=8.0.0")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)

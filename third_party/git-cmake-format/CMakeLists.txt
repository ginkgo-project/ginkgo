ginkgo_load_git_package(git-cmake-format
    "https://github.com/ginkgo-project/git-cmake-format.git"
    "2e7ebcdd65ecb507599b25b93b82d2985a8888c5"
    "-DGCF_CLANGFORMAT_MINIMAL_VERSION=5.0.1")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)

ginkgo_load_git_package(git-cmake-format
    "https://github.com/ginkgo-project/git-cmake-format.git"
    "600c509ecc00d75af5ae2c70950717787a5afe21"
    "-DGCF_CLANGFORMAT_MINIMAL_VERSION=5.0.1")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)

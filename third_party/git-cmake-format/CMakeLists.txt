ginkgo_load_git_package(git-cmake-format
    "https://github.com/ginkgo-project/git-cmake-format.git"
    "29c23665d624e1cae1308bec651706fdaa8fe38b"
    "-DGCF_CLANGFORMAT_MINIMAL_VERSION=5.0.0")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)

ginkgo_load_git_package(git-cmake-format
    "https://github.com/ginkgo-project/git-cmake-format.git"
    "e19ab13e640d58abd3bfdbff5f77b499b2ec4169")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)

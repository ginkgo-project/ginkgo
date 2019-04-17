ginkgo_load_git_package(git-cmake-format
    "https://github.com/gflegar/git-cmake-format.git"
    "9fdc1553c525b3d7ce758892fe666078903a1b21")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)

if (BUILD_TESTS)
    add_subdirectory(gtest)
endif()

if (DEVEL_TOOLS)
    add_subdirectory(git-cmake-format)
endif()

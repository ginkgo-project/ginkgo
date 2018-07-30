include(package_loader.cmake)

if(BUILD_TESTS)
    add_subdirectory(gtest)
endif()

if(DEVEL_TOOLS)
    set(GCF_IGNORE_LIST "third_party" CACHE STRING "Ignore directories for GCF")
    add_subdirectory(git-cmake-format)
endif()

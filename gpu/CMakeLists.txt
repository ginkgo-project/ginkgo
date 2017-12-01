find_package(CUDA REQUIRED)

add_subdirectory(base)


if(BUILD_TESTS)
    add_subdirectory(test)
endif()

if(NOT GTest_FOUND)
    message(STATUS "Fetching external GTest")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        6a7ed316a5cdc07b6d26362c90770787513822d4
    )
    set(gtest_disable_pthreads ON)
    if(MSVC)
        set(gtest_force_shared_crt ON)
    endif()
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
    endif()
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    # by default, the outdated targets are not being exported
    add_library(GTest::Main ALIAS gtest_main)
    add_library(GTest::GTest ALIAS gtest)
endif()

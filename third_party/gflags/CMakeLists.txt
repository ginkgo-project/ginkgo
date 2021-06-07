if(NOT gflags_FOUND)
    message(STATUS "Fetching external GFlags")
    include(FetchContent)
    FetchContent_Declare(
        gflags
        GIT_REPOSITORY https://github.com/gflags/gflags.git
        GIT_TAG        f7388c6655e699f777a5a74a3c9880b9cfaabe59
    )
    # need to set the variables in CACHE due to CMP0077
    set(gtest_disable_pthreads ON CACHE INTERNAL "")
    FetchContent_GetProperties(gflags)
    if(NOT gflags_POPULATED)
        FetchContent_Populate(gflags)
    endif()
    set(GFLAGS_BUILD_TESTING OFF)
    set(GFLAGS_BUILD_PACKAGING OFF)
    add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

message(STATUS "Fetching external GFlags")
include(FetchContent)
FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG        827c769e5fc98e0f2a34c47cef953cc6328abced
)
# need to set the variables in CACHE due to CMP0077
set(GFLAGS_BUILD_TESTING OFF CACHE INTERNAL "")
set(GFLAGS_BUILD_PACKAGING OFF CACHE INTERNAL "")
FetchContent_GetProperties(gflags)
if(NOT gflags_POPULATED)
    FetchContent_Populate(gflags)
    add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

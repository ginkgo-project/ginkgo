message(STATUS "Fetching external GFlags")
include(FetchContent)
FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG 1ad091ead028bf54d5ea68d15af343c49d18e507
)
# need to set the variables in CACHE due to CMP0077
set(GFLAGS_BUILD_TESTING OFF CACHE INTERNAL "")
set(GFLAGS_BUILD_PACKAGING OFF CACHE INTERNAL "")
FetchContent_GetProperties(gflags)
if(NOT gflags_POPULATED)
    FetchContent_Populate(gflags)
    add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

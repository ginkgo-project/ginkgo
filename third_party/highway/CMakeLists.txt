message(STATUS "Fetching external highway")
include(FetchContent)
FetchContent_Declare(
    hwy
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.3.0
)
set(HWY_ENABLE_TESTS OFF CACHE INTERNAL "")
set(HWY_ENABLE_CONTRIB OFF CACHE INTERNAL "")
set(HWY_ENABLE_EXAMPLES OFF CACHE INTERNAL "")
# Need to force static build to get the target?
set(HWY_FORCE_STATIC_LIBS ON CACHE INTERNAL "")
FetchContent_MakeAvailable(hwy)

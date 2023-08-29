message(STATUS "Fetching external nlohmann_json")
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.9.1
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install ON CACHE INTERNAL "")
# default cmake installation is in CMAKE_INSTALL_PREFIX/share/cmake
FetchContent_MakeAvailable(nlohmann_json)

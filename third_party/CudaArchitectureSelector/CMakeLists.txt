message(STATUS "Fetching external CAS")
include(FetchContent)
FetchContent_Declare(
    cuda_architecture_selector
    GIT_REPOSITORY https://github.com/ginkgo-project/CudaArchitectureSelector.git
    GIT_TAG        97022db486e25ede6d7cbf683f4ac8d48694222a
)
FetchContent_GetProperties(cuda_architecture_selector)
if(NOT cuda_architecture_selector_POPULATED)
    FetchContent_Populate(cuda_architecture_selector)
endif()
add_subdirectory("${cuda_architecture_selector_SOURCE_DIR}" "${cuda_architecture_selector_BINARY_DIR}" EXCLUDE_FROM_ALL)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

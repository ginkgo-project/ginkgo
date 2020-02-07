ginkgo_load_git_package(CudaArchitectureSelector
    "https://github.com/ginkgo-project/CudaArchitectureSelector.git"
    "51e6fc3590bfadded3bcf56f739c90066b2b8071")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

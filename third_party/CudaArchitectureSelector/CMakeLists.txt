ginkgo_load_git_package(CudaArchitectureSelector
    "https://github.com/ginkgo-project/CudaArchitectureSelector.git"
    "cba91d58e9bb847e572fedf8cc18f701ae035410")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

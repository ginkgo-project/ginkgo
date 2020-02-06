ginkgo_load_git_package(CudaArchitectureSelector
    "https://github.com/ginkgo-project/CudaArchitectureSelector.git"
    "a2d58da061c9f9ec30f20ba9ecf9959b78772c30")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

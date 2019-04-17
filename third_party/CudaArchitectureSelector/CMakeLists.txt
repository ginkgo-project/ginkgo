ginkgo_load_git_package(CudaArchitectureSelector
    "https://github.com/ginkgo-project/CudaArchitectureSelector.git"
    "0b46fb7d653404db312cbc1fc702cb528fd1c1b0")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

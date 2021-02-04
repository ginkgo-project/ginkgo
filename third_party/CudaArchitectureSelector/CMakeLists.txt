message(STATUS "Configuring and building CAS")
ginkgo_load_git_package(CudaArchitectureSelector
    "https://github.com/ginkgo-project/CudaArchitectureSelector.git"
    "97022db486e25ede6d7cbf683f4ac8d48694222a")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/build EXCLUDE_FROM_ALL)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

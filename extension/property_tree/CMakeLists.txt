add_library(property_tree INTERFACE)
target_include_directories(property_tree INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>
    )


if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

## Install step
# ginkgo_install_header_only_library(property_tree Ginkgo)
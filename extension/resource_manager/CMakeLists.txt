add_library(resource_manager INTERFACE)
target_include_directories(resource_manager INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>
    )
target_link_libraries(resource_manager INTERFACE rapidjson)

if(GINKGO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

## Install step
ginkgo_install_header_only_library(resource_manager Ginkgo)

add_library(property_tree INTERFACE)
target_include_directories(property_tree INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>
    )


if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

## Install step
install(DIRECTORY "include/property_tree"
        DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}/ginkgo/extensions"
        FILES_MATCHING PATTERN "*.hpp"
        )
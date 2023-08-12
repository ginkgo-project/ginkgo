add_library(file_config "")
add_library(file_config_custom "")

target_sources(file_config_custom
    PRIVATE
    "base/executor.cpp"
    "matrix/csr.cpp"
    "preconditioner/isai.cpp"
    "stop/iteration.cpp"
    "log/convergence.cpp"
    )


target_link_libraries(file_config_custom PUBLIC nlohmann_json::nlohmann_json Ginkgo::ginkgo)
target_include_directories(file_config_custom PUBLIC 
    $<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}/extensions/file_config/include>
    $<INSTALL_INTERFACE:include/extensions/file_config>
    )
target_sources(file_config
    PRIVATE
    "base/selection.cpp"
    )
target_link_libraries(file_config PUBLIC file_config_custom)

ginkgo_compile_features(file_config_custom)
ginkgo_compile_features(file_config)


function(ginkgo_ext_install_library name export_target)
    if (WIN32 OR CYGWIN)
        # dll is considered as runtime
        install(TARGETS "${name}"
            EXPORT "${export_target}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}"
            RUNTIME DESTINATION "${CMAKE_INSTALL_FULL_BINDIR}"
            )
    else ()
        # install .so and .a files
        install(TARGETS "${name}"
            EXPORT "${export_target}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}"
        )
    endif ()
endfunction()

ginkgo_ext_install_library(file_config_custom GkoFileConfig)
ginkgo_ext_install_library(file_config GkoFileConfig)

install(DIRECTORY "${Ginkgo_SOURCE_DIR}/extensions/file_config/include/"
        DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}/extensions/file_config"
        FILES_MATCHING PATTERN "*.hpp"
        )
# export targets
export(EXPORT GkoFileConfig
    NAMESPACE Ginkgo::
    FILE "${Ginkgo_BINARY_DIR}/extensions/file_config/GkoFileConfigTargets.cmake"
    )

# export configuration file for importing
write_basic_package_version_file(
    "${Ginkgo_BINARY_DIR}/extensions/file_config/GkoFileConfigConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
    )
configure_package_config_file(
    "${Ginkgo_SOURCE_DIR}/extensions/file_config/GkoFileConfigConfig.cmake.in"
    "${Ginkgo_BINARY_DIR}/extensions/file_config/GkoFileConfigConfig.cmake"
    INSTALL_DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
    PATH_VARS CMAKE_INSTALL_FULL_INCLUDEDIR CMAKE_INSTALL_FULL_LIBDIR CMAKE_INSTALL_PREFIX GINKGO_INSTALL_MODULE_DIR
    )
install(FILES
    "${Ginkgo_BINARY_DIR}/extensions/file_config/GkoFileConfigConfig.cmake"
    "${Ginkgo_BINARY_DIR}/extensions/file_config/GkoFileConfigConfigVersion.cmake"
    DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
    )
install(EXPORT Ginkgo
    NAMESPACE Ginkgo::
    FILE GkoFileConfigTargets.cmake
    DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}")
if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

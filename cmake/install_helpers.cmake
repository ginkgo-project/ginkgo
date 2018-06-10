include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

function(ginkgo_install_library name subdir)
    # install .so and .a files
    install(TARGETS "${name}"
        EXPORT Ginkgo
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        )
    # copy header files
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/${subdir}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp"
        )
endfunction()

function(ginkgo_default_includes name)
    # set include path depending on used interface
    target_include_directories("${name}"
        PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        )
endfunction()

function(ginkgo_install)
    # export targets
    set(INSTALL_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/Ginkgo")
    install(EXPORT Ginkgo
        FILE GinkgoTargets.cmake
        NAMESPACE Ginkgo::
        DESTINATION "${INSTALL_CONFIG_DIR}"
        )

    # export configuration file for importing
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/GinkgoConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
        )
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/GinkgoConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/GinkgoConfig.cmake"
        INSTALL_DESTINATION "${INSTALL_CONFIG_DIR}"
        )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/GinkgoConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/GinkgoConfigVersion.cmake"
        DESTINATION "${INSTALL_CONFIG_DIR}"
        )
endfunction()

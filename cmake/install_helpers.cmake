include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

function(ginkgo_install_library name subdir)
    # install .so and .a files
    install(TARGETS "${name}"
        EXPORT Ginkgo
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
endfunction()

function(ginkgo_install)
    # install the public header files
    install(DIRECTORY "${Ginkgo_SOURCE_DIR}/include/"
        DESTINATION "include"
        FILES_MATCHING PATTERN "*.hpp"
        )
    install(DIRECTORY "${Ginkgo_BINARY_DIR}/include/"
        DESTINATION "include"
        FILES_MATCHING PATTERN "*.hpp"
        )

    # export targets
    set(INSTALL_CONFIG_DIR "lib/cmake/Ginkgo")
    install(EXPORT Ginkgo
        FILE GinkgoTargets.cmake
        NAMESPACE Ginkgo::
        DESTINATION "${INSTALL_CONFIG_DIR}"
        )

    # export configuration file for importing
    write_basic_package_version_file(
        "${Ginkgo_BINARY_DIR}/GinkgoConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
        )
    configure_package_config_file(
        "${Ginkgo_SOURCE_DIR}/cmake/GinkgoConfig.cmake.in"
        "${Ginkgo_BINARY_DIR}/GinkgoConfig.cmake"
        INSTALL_DESTINATION "${INSTALL_CONFIG_DIR}"
        )
    install(FILES
        "${Ginkgo_BINARY_DIR}/GinkgoConfig.cmake"
        "${Ginkgo_BINARY_DIR}/GinkgoConfigVersion.cmake"
        DESTINATION "${INSTALL_CONFIG_DIR}"
        )

    # Export package for use from the build tree
    if (GINKGO_EXPORT_BUILD_DIR)
        export(PACKAGE Ginkgo)
    endif()
endfunction()

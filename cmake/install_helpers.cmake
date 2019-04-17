include(CMakePackageConfigHelpers)
include(GNUInstallDirs)


set(GINKGO_INSTALL_INCLUDE_DIR "include")
set(GINKGO_INSTALL_LIBRARY_DIR "lib")
set(GINKGO_INSTALL_PKGCONFIG_DIR "lib/pkgconfig")
set(GINKGO_INSTALL_CONFIG_DIR "lib/cmake/Ginkgo")

function(ginkgo_install_library name subdir)
    # install .so and .a files
    install(TARGETS "${name}"
        EXPORT Ginkgo
        LIBRARY DESTINATION ${GINKGO_INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${GINKGO_INSTALL_LIBRARY_DIR}
        )
endfunction()

function(ginkgo_install)
    # pkg-config file
    install(FILES "${Ginkgo_BINARY_DIR}/ginkgo.pc" DESTINATION "${GINKGO_INSTALL_PKGCONFIG_DIR}")

    # install the public header files
    install(DIRECTORY "${Ginkgo_SOURCE_DIR}/include/"
        DESTINATION "${GINKGO_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING PATTERN "*.hpp"
        )
    install(DIRECTORY "${Ginkgo_BINARY_DIR}/include/"
        DESTINATION "${GINKGO_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING PATTERN "*.hpp"
        )
    if (GINKGO_HAVE_PAPI_SDE)
        install(FILES "${Ginkgo_SOURCE_DIR}/third_party/papi_sde/papi_sde_interface.h"
            DESTINATION "${GINKGO_INSTALL_INCLUDE_DIR}/third_party/papi_sde"
            )
    endif()

    # export targets
    export(EXPORT Ginkgo
        NAMESPACE Ginkgo::
        FILE "${Ginkgo_BINARY_DIR}/GinkgoTargets.cmake"
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
        INSTALL_DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
        )
    install(FILES
        "${Ginkgo_BINARY_DIR}/GinkgoConfig.cmake"
        "${Ginkgo_BINARY_DIR}/GinkgoConfigVersion.cmake"
        "${Ginkgo_BINARY_DIR}/GinkgoTargets.cmake"
        DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
        )

    # Export package for use from the build tree
    if (GINKGO_EXPORT_BUILD_DIR)
        export(PACKAGE Ginkgo)
    endif()
endfunction()

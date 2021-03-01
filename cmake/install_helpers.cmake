include(CMakePackageConfigHelpers)
include(GNUInstallDirs)


set(GINKGO_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(GINKGO_INSTALL_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}")
set(GINKGO_INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
set(GINKGO_INSTALL_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/Ginkgo")
set(GINKGO_INSTALL_MODULE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/Ginkgo/Modules")

function(ginkgo_install_library name subdir)

    if (WIN32 OR CYGWIN)
        # dll is considered as runtime
        install(TARGETS "${name}"
            EXPORT Ginkgo
            LIBRARY DESTINATION "${GINKGO_INSTALL_LIBRARY_DIR}"
            ARCHIVE DESTINATION "${GINKGO_INSTALL_LIBRARY_DIR}"
            RUNTIME DESTINATION "${GINKGO_INSTALL_LIBRARY_DIR}"
            )
    else ()
        # install .so and .a files
        install(TARGETS "${name}"
            EXPORT Ginkgo
            LIBRARY DESTINATION "${GINKGO_INSTALL_LIBRARY_DIR}"
            ARCHIVE DESTINATION "${GINKGO_INSTALL_LIBRARY_DIR}"
        )
    endif ()
endfunction()

function(ginkgo_install)
    # pkg-config file
    install(FILES "${Ginkgo_BINARY_DIR}/ginkgo.pc" DESTINATION "${GINKGO_INSTALL_PKGCONFIG_DIR}")

    # install the public header files
    install(DIRECTORY "${Ginkgo_SOURCE_DIR}/include/"
        DESTINATION "${GINKGO_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING PATTERN "*.hpp"
        )
    install(FILES "${Ginkgo_BINARY_DIR}/include/ginkgo/config.hpp"
        DESTINATION "${GINKGO_INSTALL_INCLUDE_DIR}/ginkgo"
        )
    if (GINKGO_HAVE_PAPI_SDE)
        install(FILES "${Ginkgo_SOURCE_DIR}/third_party/papi_sde/papi_sde_interface.h"
            DESTINATION "${GINKGO_INSTALL_INCLUDE_DIR}/third_party/papi_sde"
            )
        install(FILES "${Ginkgo_SOURCE_DIR}/cmake/Modules/FindPAPI.cmake"
            DESTINATION "${GINKGO_INSTALL_MODULE_DIR}/"
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
        COMPATIBILITY SameMajorVersion
        )
    configure_package_config_file(
        "${Ginkgo_SOURCE_DIR}/cmake/GinkgoConfig.cmake.in"
        "${Ginkgo_BINARY_DIR}/GinkgoConfig.cmake"
        INSTALL_DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
        )
    set(HELPERS "hip_helpers.cmake" "windows_helpers.cmake")
    foreach (helper ${HELPERS})
        configure_file(${Ginkgo_SOURCE_DIR}/cmake/${helper}
            ${Ginkgo_BINARY_DIR}/${helper} COPYONLY)
    endforeach()
    install(FILES
        "${Ginkgo_BINARY_DIR}/GinkgoConfig.cmake"
        "${Ginkgo_BINARY_DIR}/GinkgoConfigVersion.cmake"
        DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
        )
    if (WIN32 OR CYGWIN)
        install(FILES
            "${Ginkgo_SOURCE_DIR}/cmake/windows_helpers.cmake"
            DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
            )
    endif()
    if (GINKGO_BUILD_HIP)
        install(FILES
            "${Ginkgo_SOURCE_DIR}/cmake/hip_helpers.cmake"
            DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
            )
    endif()
    install(EXPORT Ginkgo
        NAMESPACE Ginkgo::
        FILE GinkgoTargets.cmake
        DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}")

    # Export package for use from the build tree
    if (GINKGO_EXPORT_BUILD_DIR)
        export(PACKAGE Ginkgo)
    endif()
endfunction()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)


set(GINKGO_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(GINKGO_INSTALL_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}")
set(GINKGO_INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
set(GINKGO_INSTALL_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/Ginkgo")
set(GINKGO_INSTALL_MODULE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/Ginkgo/Modules")

# This function adds the correct RPATH properties to a Ginkgo target.
#
# The behavior depends on three options GINKGO_INSTALL_RPATH[*] variables. It
# does the following:
#
# 1. GINKGO_INSTALL_RPATH : If this flag is not set, no RPATH information is
#    added.
# 2. GINKGO_INSTALL_RPATH_ORIGIN : Allows adding the library directory to the
#    RPATH.
# 3. GINKGO_INSTALL_RPATH_DEPENDENCIES : Allows adding any extra paths to the
#    RPATH.
#
# @param name  the name of the target
# @param ARGN  any external dependencies path to be added
function(ginkgo_add_install_rpath name)
    if (GINKGO_INSTALL_RPATH_ORIGIN)
        if (APPLE)
            set(ORIGIN_OR_LOADER_PATH "@loader_path")
        else()
            set(ORIGIN_OR_LOADER_PATH "$ORIGIN")
        endif()
    endif()
    if (GINKGO_INSTALL_RPATH_DEPENDENCIES)
        set(RPATH_DEPENDENCIES "${ARGN}")
        if  (GINKGO_HAVE_HWLOC AND GINKGO_USE_EXTERNAL_HWLOC)
            get_filename_component(HWLOC_LIB_PATH ${HWLOC_LIBRARIES} DIRECTORY)
            list(APPEND RPATH_DEPENDENCIES "${HWLOC_LIBRARIES}")
        endif()
    endif()
    if (GINKGO_INSTALL_RPATH)
        set_property(TARGET "${name}" PROPERTY INSTALL_RPATH
            "${ORIGIN_OR_LOADER_PATH}" "${RPATH_DEPENDENCIES}")
    endif()
endfunction()

# Handles installation settings for a Ginkgo library.
#
# @param name  the name of the Ginkgo library target
# @param ARGN  this should contain any external dependency's library PATH
function(ginkgo_install_library name)
    ginkgo_add_install_rpath("${name}" "${ARGN}")

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

    if  (GINKGO_HAVE_HWLOC AND NOT GINKGO_USE_EXTERNAL_HWLOC)
        get_filename_component(HWLOC_LIB_PATH ${HWLOC_LIBRARIES} DIRECTORY)
        file(GLOB HWLOC_LIBS "${HWLOC_LIB_PATH}/libhwloc*")
        install(FILES ${HWLOC_LIBS}
            DESTINATION "${GINKGO_INSTALL_LIBRARY_DIR}"
            )
        # We only use hwloc and not netloc
        install(DIRECTORY "${HWLOC_INCLUDE_DIRS}/hwloc"
            DESTINATION "${GINKGO_INSTALL_INCLUDE_DIR}"
            )
        install(FILES "${HWLOC_INCLUDE_DIRS}/hwloc.h"
            DESTINATION "${GINKGO_INSTALL_INCLUDE_DIR}"
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
    set(HELPERS "windows_helpers.cmake")
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
    install(EXPORT Ginkgo
        NAMESPACE Ginkgo::
        FILE GinkgoTargets.cmake
        DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}")

    # Export package for use from the build tree
    if (GINKGO_EXPORT_BUILD_DIR)
        export(PACKAGE Ginkgo)
    endif()
endfunction()

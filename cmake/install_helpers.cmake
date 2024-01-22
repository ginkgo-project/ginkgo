include(CMakePackageConfigHelpers)
include(GNUInstallDirs)


set(GINKGO_INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig")
set(GINKGO_INSTALL_CONFIG_DIR "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/Ginkgo")
set(GINKGO_INSTALL_MODULE_DIR "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/Ginkgo/Modules")

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

    install(TARGETS "${name}"
        EXPORT Ginkgo
        LIBRARY
            DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}"
            COMPONENT Ginkgo_Runtime
            NAMELINK_COMPONENT Ginkgo_Development
        RUNTIME
            DESTINATION "${CMAKE_INSTALL_FULL_BINDIR}"
            COMPONENT Ginkgo_Runtime
        ARCHIVE
            DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}"
            COMPONENT Ginkgo_Development
        )
endfunction()

function(ginkgo_install)
    # pkg-config file
    install(FILES ${Ginkgo_BINARY_DIR}/ginkgo_$<CONFIG>.pc
        DESTINATION "${GINKGO_INSTALL_PKGCONFIG_DIR}"
        RENAME ginkgo.pc
        COMPONENT Ginkgo_Development)

    # install the public header files
    install(DIRECTORY "${Ginkgo_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}"
        COMPONENT Ginkgo_Development
        FILES_MATCHING PATTERN "*.hpp"
        )
    install(FILES "${Ginkgo_BINARY_DIR}/include/ginkgo/config.hpp"
        DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}/ginkgo"
        COMPONENT Ginkgo_Development
        )

    if  (GINKGO_HAVE_HWLOC AND NOT HWLOC_FOUND)
        get_filename_component(HWLOC_LIB_PATH ${HWLOC_LIBRARIES} DIRECTORY)
        file(GLOB HWLOC_LIBS "${HWLOC_LIB_PATH}/libhwloc*")
        install(FILES ${HWLOC_LIBS}
            DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}"
            COMPONENT Ginkgo_Runtime
            )
        # We only use hwloc and not netloc
        install(DIRECTORY "${HWLOC_INCLUDE_DIRS}/hwloc"
            DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}"
            COMPONENT Ginkgo_Development
            )
        install(FILES "${HWLOC_INCLUDE_DIRS}/hwloc.h"
            DESTINATION "${CMAKE_INSTALL_FULL_INCLUDEDIR}"
            COMPONENT Ginkgo_Development
            )
    endif()

    # Install CMake modules
    install(DIRECTORY "${Ginkgo_SOURCE_DIR}/cmake/Modules/"
        DESTINATION "${GINKGO_INSTALL_MODULE_DIR}"
        COMPONENT Ginkgo_Development
        FILES_MATCHING PATTERN "*.cmake"
        )

    set(GINKGO_EXPORT_BINARY_DIR OFF)

    # export configuration file for importing
    write_basic_package_version_file(
        "${Ginkgo_BINARY_DIR}/cmake/GinkgoConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
        )
    configure_package_config_file(
        "${Ginkgo_SOURCE_DIR}/cmake/GinkgoConfig.cmake.in"
        "${Ginkgo_BINARY_DIR}/cmake/GinkgoConfig.cmake"
        INSTALL_DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
        PATH_VARS CMAKE_INSTALL_FULL_INCLUDEDIR CMAKE_INSTALL_FULL_LIBDIR CMAKE_INSTALL_PREFIX GINKGO_INSTALL_MODULE_DIR
        )
    install(FILES
        "${Ginkgo_BINARY_DIR}/cmake/GinkgoConfig.cmake"
        "${Ginkgo_BINARY_DIR}/cmake/GinkgoConfigVersion.cmake"
        DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
        COMPONENT Ginkgo_Development)
    install(EXPORT Ginkgo
        NAMESPACE Ginkgo::
        FILE GinkgoTargets.cmake
        DESTINATION "${GINKGO_INSTALL_CONFIG_DIR}"
        COMPONENT Ginkgo_Development)

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BUILD_SHARED_LIBS)
        install(FILES
            "${Ginkgo_SOURCE_DIR}/dev_tools/scripts/gdb-ginkgo.py"
            DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}"
            RENAME "$<TARGET_FILE_NAME:ginkgo>-gdb.py"
            COMPONENT Ginkgo_Development)
    endif()
endfunction()


function(ginkgo_export_binary_dir)
    # export targets
    export(EXPORT Ginkgo
           NAMESPACE Ginkgo::
           FILE "${Ginkgo_BINARY_DIR}/GinkgoTargets.cmake"
    )

    set(GINKGO_EXPORT_BINARY_DIR ON)
    set(GINKGO_INSTALL_MODULE_DIR "${Ginkgo_SOURCE_DIR}/cmake/Modules/")

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
      PATH_VARS GINKGO_INSTALL_MODULE_DIR
      INSTALL_PREFIX ${Ginkgo_BINARY_DIR}
    )

    # Export package for use from the build tree
    export(PACKAGE Ginkgo)
endfunction()

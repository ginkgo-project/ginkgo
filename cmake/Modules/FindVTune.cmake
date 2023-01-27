#.rst:
# FindROCTX
# -------
#
# Find the VTune ITT headers and library.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``VTune::ITT``
#   The VTune ITT library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``VTune_INCLUDE_DIRS``
#   where to find ittnotify.h
#
# ``VTune_LIBRARIES``
#   the libraries to link against in order to use the VTune ITT library.
#
# ``VTune_FOUND``
#   If false, do not try to use the VTune library.

find_path(VTune_INCLUDE_DIR NAMES ittnotify.h HINTS ${VTUNE_PATH}/include)
mark_as_advanced(VTune_INCLUDE_DIR)

if(NOT VTune_LIBRARY)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        find_library(VTune_LIBRARY NAMES ittnotify HINTS ${VTUNE_PATH}/lib64)
    else()
        find_library(VTune_LIBRARY NAMES ittnotify HINTS ${VTUNE_PATH}/lib32)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VTune REQUIRED_VARS VTune_LIBRARY VTune_INCLUDE_DIR)

if(VTune_FOUND)
    set(VTune_LIBRARIES ${VTune_LIBRARY})
    set(VTune_INCLUDE_DIRS ${VTune_INCLUDE_DIR})
    unset(VTune_LIBRARY)
    unset(VTune_INCLUDE_DIR)

    if(NOT TARGET VTune::ITT)
        add_library(VTune::ITT UNKNOWN IMPORTED)
        set_target_properties(VTune::ITT PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${VTune_INCLUDE_DIRS}")
        if(EXISTS "${VTune_LIBRARIES}")
            set_target_properties(VTune::ITT PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${VTune_LIBRARIES}")
        endif()
    endif()
endif()

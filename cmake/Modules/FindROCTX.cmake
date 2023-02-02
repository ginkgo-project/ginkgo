#.rst:
# FindROCTX
# -------
#
# Find the ROCTX headers and library, usually provided by ROCm.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``roc::roctx``
#   The ROCTX library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``ROCTX_INCLUDE_DIRS``
#   where to find roctx.h
#
# ``ROCTX_LIBRARIES``
#   the libraries to link against in order to use the ROCTX library.
#
# ``ROCTX_FOUND``
#   If false, do not try to use the ROCTX library.

find_path(ROCTX_INCLUDE_DIR NAMES roctx.h HINTS ${ROCTRACER_PATH}/include)
mark_as_advanced(ROCTX_INCLUDE_DIR)

if(NOT ROCTX_LIBRARY)
    find_library(ROCTX_LIBRARY NAMES roctx64 HINTS ${ROCTRACER_PATH}/lib)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCTX REQUIRED_VARS ROCTX_LIBRARY ROCTX_INCLUDE_DIR)

if(ROCTX_FOUND)
    set(ROCTX_LIBRARIES ${ROCTX_LIBRARY})
    set(ROCTX_INCLUDE_DIRS ${ROCTX_INCLUDE_DIR})
    unset(ROCTX_LIBRARY)
    unset(ROCTX_INCLUDE_DIR)

    if(NOT TARGET roc::roctx)
        add_library(roc::roctx UNKNOWN IMPORTED)
        set_target_properties(roc::roctx PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ROCTX_INCLUDE_DIRS}")
        if(EXISTS "${ROCTX_LIBRARIES}")
            set_target_properties(roc::roctx PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${ROCTX_LIBRARIES}")
        endif()
    endif()
endif()

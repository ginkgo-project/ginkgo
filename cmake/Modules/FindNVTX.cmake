#.rst:
# FindNVTX
# -------
#
# Find the NVTX headers and library, usually provided by CUDA.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``nvtx::nvtx``
#   The NVTX library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``NVTX_INCLUDE_DIRS``
#   where to find nvToolsExt.h
#
# ``NVTX_LIBRARIES``
#   the libraries to link against in order to use the NVTX library.
#
# ``NVTX_FOUND``
#   If false, do not try to use the ROCTX library.

find_path(NVTX_INCLUDE_DIR NAMES nvToolsExt.h HINTS ${CUDA_INCLUDE_DIRS})
mark_as_advanced(NVTX_INCLUDE_DIR)

if(NOT NVTX_LIBRARY)
    find_library(NVTX_LIBRARY NAMES nvToolsExt HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX REQUIRED_VARS NVTX_LIBRARY NVTX_INCLUDE_DIR)

if(NVTX_FOUND)
    set(NVTX_LIBRARIES ${NVTX_LIBRARY})
    set(NVTX_INCLUDE_DIRS ${NVTX_INCLUDE_DIR})
    unset(NVTX_LIBRARY)
    unset(NVTX_INCLUDE_DIR)

    if(NOT TARGET nvtx::nvtx)
        add_library(nvtx::nvtx UNKNOWN IMPORTED)
        set_target_properties(nvtx::nvtx PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NVTX_INCLUDE_DIRS}")
        if(EXISTS "${NVTX_LIBRARIES}")
            set_target_properties(nvtx::nvtx PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${NVTX_LIBRARIES}")
        endif()
    endif()
endif()

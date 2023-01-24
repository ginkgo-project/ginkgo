#.rst:
# FindNVTX
# -------
#
# Find the NVTX headers (and potentially library), usually provided by CUDA.
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
#   If false, do not try to use the NVTX library.

find_path(NVTX3_INCLUDE_DIR NAMES nvToolsExt.h HINTS ${CUDA_INCLUDE_DIRS}/nvtx3)
find_path(NVTX_INCLUDE_DIR NAMES nvToolsExt.h HINTS ${CUDA_INCLUDE_DIRS})
mark_as_advanced(NVTX3_INCLUDE_DIR)
mark_as_advanced(NVTX_INCLUDE_DIR)
include(FindPackageHandleStandardArgs)

if(NOT NVTX3_INCLUDE_DIR)
    find_library(NVTX_LIBRARY NAMES nvToolsExt HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    mark_as_advanced(NVTX_LIBRARY)
    find_package_handle_standard_args(NVTX REQUIRED_VARS NVTX_LIBRARY NVTX_INCLUDE_DIR)
else()
    find_package_handle_standard_args(NVTX REQUIRED_VARS NVTX3_INCLUDE_DIR)
endif()

if(NVTX_FOUND)
    if(NOT NVTX3_INCLUDE_DIR)
        set(NVTX_INCLUDE_DIRS ${NVTX_INCLUDE_DIR})
        set(NVTX_LIBRARIES ${NVTX_LIBRARY})
        if(NOT TARGET nvtx::nvtx)
            add_library(nvtx::nvtx UNKNOWN IMPORTED)
            set_target_properties(nvtx::nvtx PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NVTX_INCLUDE_DIRS}")
            set_target_properties(nvtx::nvtx PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS GKO_LEGACY_NVTX)
            if(EXISTS "${NVTX_LIBRARIES}")
                set_target_properties(nvtx::nvtx PROPERTIES
                    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                    IMPORTED_LOCATION "${NVTX_LIBRARIES}")
            endif()
        endif()
    else()
        if(NOT TARGET nvtx::nvtx)            
            set(NVTX_INCLUDE_DIRS ${NVTX3_INCLUDE_DIR}/..)
            add_library(nvtx::nvtx INTERFACE IMPORTED)
            set_target_properties(nvtx::nvtx PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NVTX_INCLUDE_DIRS}")
            if(WIN32)
                # handle Windows.h defining min/max macros
                set_target_properties(nvtx::nvtx PROPERTIES
                    INTERFACE_COMPILE_DEFINITIONS NOMINMAX)
            endif()
        endif()
    endif()
    unset(NVTX_LIBRARY)
    unset(NVTX_INCLUDE_DIR)
    unset(NVTX3_INCLUDE_DIR)    
endif()

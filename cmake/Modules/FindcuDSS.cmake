#.rst:
# FindcuDSS
# ---------
#
# Find the NVIDIA cuDSS (CUDA Direct Sparse Solver) library.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``cuDSS::cuDSS``
#   The cuDSS library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``CUDSS_INCLUDE_DIR``
#   where to find cudss.h
#
# ``CUDSS_LIBRARY``
#   the library to link against in order to use cuDSS.
#
# ``cuDSS_FOUND``
#   If false, do not try to use the cuDSS library.
#
# Hints
# ^^^^^
#
# ``CUDSS_DIR``
#   Set this variable or the corresponding environment variable to a cuDSS
#   installation prefix to help locate it.

include(FindPackageHandleStandardArgs)

find_path(
    CUDSS_INCLUDE_DIR
    NAMES cudss.h
    HINTS ${CUDSS_DIR} $ENV{CUDSS_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include
)

find_library(
    CUDSS_LIBRARY
    NAMES cudss
    HINTS ${CUDSS_DIR} $ENV{CUDSS_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
    cuDSS
    REQUIRED_VARS CUDSS_LIBRARY CUDSS_INCLUDE_DIR
)

if(cuDSS_FOUND AND NOT TARGET cuDSS::cuDSS)
    add_library(cuDSS::cuDSS UNKNOWN IMPORTED)
    set_target_properties(
        cuDSS::cuDSS
        PROPERTIES
            IMPORTED_LOCATION "${CUDSS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(CUDSS_INCLUDE_DIR CUDSS_LIBRARY)

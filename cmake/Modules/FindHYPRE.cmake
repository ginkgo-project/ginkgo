#.rst:
# FindHYPRE
# -------
#
# Find the HYPRE library (high performance preconditioners).
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``HYPRE::HYPRE``
#   The HYPRE library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``HYPRE_INCLUDE_DIRS``
#   where to find HYPRE.h, HYPRE_parcsr_ls.h, etc.
#
# ``HYPRE_LIBRARIES``
#   the libraries to link against in order to use HYPRE.
#
# ``HYPRE_FOUND``
#   If false, do not try to use the HYPRE library.

find_path(
    HYPRE_INCLUDE_DIR
    NAMES HYPRE.h
    HINTS ${HYPRE_DIR}
    ENV HYPRE_DIR
    PATH_SUFFIXES include
)

find_library(
    HYPRE_LIBRARY
    NAMES HYPRE
    HINTS ${HYPRE_DIR}
    ENV HYPRE_DIR
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    HYPRE
    REQUIRED_VARS HYPRE_LIBRARY HYPRE_INCLUDE_DIR
)

if(HYPRE_FOUND)
    set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})
    set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})
    unset(HYPRE_INCLUDE_DIR)
    unset(HYPRE_LIBRARY)

    if(NOT TARGET HYPRE::HYPRE)
        add_library(HYPRE::HYPRE INTERFACE IMPORTED)
        set_target_properties(
            HYPRE::HYPRE
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${HYPRE_LIBRARIES}"
        )
    endif()
endif()

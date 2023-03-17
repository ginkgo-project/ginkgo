#.rst:
# FindMETIS
# -------
#
# Find the METIS graph partitioning library.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``METIS::METIS``
#   The METIS library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``METIS_INCLUDE_DIRS``
#   where to find metis.h or metis64.h
#
# ``METIS_HEADER``
#   either metis64.h or metis.h, dependent on which was found
#
# ``METIS_LIBRARIES``
#   the libraries to link against in order to use the METIS library.
#
# ``METIS_FOUND``
#   If false, do not try to use the METIS library.

find_path(METIS_INCLUDE_DIR NAMES metis64.h metis.h HINTS ${METIS_DIR} ENV METIS_DIR PATH_SUFFIXES include)

if (METIS_INCLUDE_DIR)
    if (EXISTS ${METIS_INCLUDE_DIR}/metis64.h)
        set(METIS_HEADER metis64.h)
        set(METIS_LIB_NAME metis64)
    else()
        set(METIS_HEADER metis.h)
        set(METIS_LIB_NAME metis)
    endif()
    file(STRINGS ${METIS_INCLUDE_DIR}/${METIS_HEADER} metis_version_str_major REGEX "^#define[\t ]+METIS_VER_MAJOR[\t ]+.*")
    file(STRINGS ${METIS_INCLUDE_DIR}/${METIS_HEADER} metis_version_str_minor REGEX "^#define[\t ]+METIS_VER_MINOR[\t ]+.*")
    string(REGEX REPLACE "^#define[\t ]+METIS_VER_MAJOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_MAJOR "${metis_version_str_major}")
    string(REGEX REPLACE "^#define[\t ]+METIS_VER_MINOR[\t ]+([0-9]+).*" "\\1" METIS_VERSION_MINOR "${metis_version_str_minor}")
    set(METIS_VERSION ${METIS_VERSION_MAJOR}.${METIS_VERSION_MINOR})
    find_library(METIS_LIBRARY ${METIS_LIB_NAME} HINTS ${METIS_DIR} ENV METIS_DIR PATH_SUFFIXES lib lib64)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR VERSION_VAR METIS_VERSION)

if(METIS_FOUND)
    set(METIS_LIBRARIES ${METIS_LIBRARY})
    set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
    unset(METIS_LIBRARY)
    unset(METIS_INCLUDE_DIR)

    if(NOT TARGET METIS::METIS)
        add_library(METIS::METIS UNKNOWN IMPORTED)
        set_target_properties(METIS::METIS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIRS}")
        set_target_properties(METIS::METIS PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${METIS_LIBRARIES}")
    endif()
endif()

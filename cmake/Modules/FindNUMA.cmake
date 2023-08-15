# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

#.rst:
# FindNUMA
# -------
#
# Find the NUMA library, usually provided by `numactl`.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``NUMA::NUMA``
#   The NUMA library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``NUMA_INCLUDE_DIRS``
#   where to find numa.h
#
# ``NUMA_LIBRARIES``
#   the libraries to link against in order to use the NUMA library.
#
# ``NUMA_FOUND``
#   If false, do not try to use the NUMA library.


find_path(NUMA_ROOT_DIR NAMES include/numa.h)

find_path(NUMA_INCLUDE_DIR NAMES numa.h HINTS ${NUMA_ROOT_DIR})
mark_as_advanced(NUMA_INCLUDE_DIR)


if(NOT NUMA_LIBRARY)
    find_library(NUMA_LIBRARY NAMES numa HINTS ${NUMA_ROOT_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA REQUIRED_VARS NUMA_LIBRARY NUMA_INCLUDE_DIR)

if(NUMA_FOUND)
    set(NUMA_LIBRARIES ${NUMA_LIBRARY})
    set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR})
    unset(NUMA_LIBRARY)
    unset(NUMA_INCLUDE_DIR)

    if(NOT TARGET NUMA::NUMA)
        add_library(NUMA::NUMA UNKNOWN IMPORTED)
        set_target_properties(NUMA::NUMA PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIRS}")
        if(EXISTS "${NUMA_LIBRARIES}")
            set_target_properties(NUMA::NUMA PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${NUMA_LIBRARIES}")
        endif()
    endif()
endif()

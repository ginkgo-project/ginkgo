# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#[=======================================================================[.rst:
FindParMETIS
------------

Find Parallel Graph Partitioning library
`ParMETIS <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview>`_

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following `IMPORTED`_ target:

``ParMETIS::ParMETIS``
  The libraries, flags, and includes to use for ParMETIS, if found.

.. _IMPORTED: https://cmake.org/cmake/help/latest/command/add_library.html#id5

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ParMETIS_FOUND``
  The ParMETIS library with all its dependencies is found

Cache Variables
^^^^^^^^^^^^^^^

The following variables may be set to influence this module's behavior:

``PARMETIS_INCLUDE_DIR``
  Include directory where the parmetis.h is found.

``PARMETIS_LIBRARY``
  Full path to the ParMETIS library

#]=======================================================================]

# text for feature summary
include(FeatureSummary)
set_package_properties(
    "ParMETIS"
    PROPERTIES
        DESCRIPTION "Parallel Graph Partitioning"
        URL "http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview"
)

find_path(PARMETIS_INCLUDE_DIR parmetis.h PATH_SUFFIXES parmetis)

# determine version of ParMETIS installation
find_file(
    PARMETIS_HEADER_FILE
    parmetis.h
    PATHS ${PARMETIS_INCLUDE_DIR}
    NO_DEFAULT_PATH
)

if(PARMETIS_HEADER_FILE)
    file(READ "${PARMETIS_HEADER_FILE}" parmetisheader)
    string(
        REGEX REPLACE
        ".*#define PARMETIS_MAJOR_VERSION[ ]+([0-9]+).*"
        "\\1"
        ParMETIS_MAJOR_VERSION
        "${parmetisheader}"
    )
    string(
        REGEX REPLACE
        ".*#define PARMETIS_MINOR_VERSION[ ]+([0-9]+).*"
        "\\1"
        ParMETIS_MINOR_VERSION
        "${parmetisheader}"
    )
    if(
        ParMETIS_MAJOR_VERSION GREATER_EQUAL 0
        AND ParMETIS_MINOR_VERSION GREATER_EQUAL 0
    )
        set(ParMETIS_VERSION
            "${ParMETIS_MAJOR_VERSION}.${ParMETIS_MINOR_VERSION}"
        )
    endif()

    set(PARMETIS_HEADER parmetis.h)
endif()
unset(PARMETIS_HEADER_FILE CACHE)

# search ParMETIS library
if(IS_PTSCOTCH_PARMETIS_HEADER)
    find_library(PARMETIS_LIBRARY ptscotchparmetis)
else()
    find_library(PARMETIS_LIBRARY parmetis)
endif()

mark_as_advanced(PARMETIS_INCLUDE_DIR PARMETIS_LIBRARY)

# minimal requires METIS version 5.0 for ParMETIS >= 4.0
if(ParMETIS_VERSION VERSION_GREATER_EQUAL "4.0")
    set(METIS_MIN_VERSION "5.0")
endif()

# find package dependencies first
find_package(MPI)
find_package(METIS ${METIS_MIN_VERSION})

# set a list of required dependencies for ParMETIS
set(PARMETIS_DEPENDENCIES METIS_FOUND MPI_FOUND)

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    "ParMETIS"
    REQUIRED_VARS PARMETIS_LIBRARY PARMETIS_INCLUDE_DIR ${PARMETIS_DEPENDENCIES}
    VERSION_VAR ParMETIS_VERSION
)

# create imported target ParMETIS::ParMETIS
if(PARMETIS_FOUND AND NOT TARGET ParMETIS::ParMETIS)
    add_library(ParMETIS::ParMETIS UNKNOWN IMPORTED)
    set_target_properties(
        ParMETIS::ParMETIS
        PROPERTIES
            IMPORTED_LOCATION ${PARMETIS_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${PARMETIS_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES "METIS::METIS;MPI::MPI_CXX"
    )
endif()

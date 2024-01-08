###
#
# @copyright (c) 2012-2020 Inria. All rights reserved.
# @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
#
# Copyright 2012-2013 Emmanuel Agullo
# Copyright 2012-2013 Mathieu Faverge
# Copyright 2012      Cedric Castagnede
# Copyright 2013-2020 Florent Pruvost
# Copyright 2020 - 2024 Ginkgo Project
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file MORSE-Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of Morse, substitute the full
#  License text for the above reference.)
#
# Modified for Ginkgo (See ABOUT-LICENSING.md for additional details).
#
###
#
# - Find HWLOC include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(HWLOC
#               [REQUIRED] [VERSION]) # Fail with error if hwloc is not found
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#   ``hwloc``
#
#=============================================================================
include(CheckStructHasMember)
include(CheckCSourceCompiles)

include(hwloc_helpers)

find_path(HWLOC_INCLUDE_DIRS
    NAMES "hwloc.h"
    HINTS ${HWLOC_DIR} $ENV{HWLOC_DIR}
    PATH_SUFFIXES include
    DOC "Find the hwloc.h main header"
    )

find_library(HWLOC_LIBRARIES "hwloc"
    HINTS ${HWLOC_DIR} $ENV{HWLOC_DIR}
    PATH_SUFFIXES lib lib64
    DOC "Find the hwloc library"
    )

if (HWLOC_INCLUDE_DIRS)
    # Find the version of hwloc found
    if(NOT HWLOC_VERSION)
        file(READ "${HWLOC_INCLUDE_DIRS}/hwloc.h"
            HEADER_CONTENTS LIMIT 16384)
        string(REGEX REPLACE ".*#define HWLOC_API_VERSION (0[xX][0-9a-fA-F]+).*" "\\1"
            HWLOC_API_VERSION "${HEADER_CONTENTS}")
        string(SUBSTRING "${HWLOC_API_VERSION}" 4 2 HEX_MAJOR)
        string(SUBSTRING "${HWLOC_API_VERSION}" 6 2 HEX_MINOR)
        string(SUBSTRING "${HWLOC_API_VERSION}" 8 2 HEX_PATCH)
        get_dec_from_hex("${HEX_MAJOR}" DEC_MAJOR)
        get_dec_from_hex("${HEX_MINOR}" DEC_MINOR)
        get_dec_from_hex("${HEX_PATCH}" DEC_PATCH)
        set(HWLOC_VERSION "${DEC_MAJOR}.${DEC_MINOR}.${DEC_PATCH}" CACHE STRING "HWLOC version")
    endif()
endif()

if (NOT HWLOC_FIND_QUIETLY)
    if (HWLOC_INCLUDE_DIRS AND HWLOC_LIBRARIES)
        message(STATUS "Looking for HWLOC - found version ${HWLOC_VERSION}")
    else()
        message(STATUS "${Magenta}Looking for HWLOC - not found"
            "\n   Please check that your environment variable HWLOC_DIR"
            "\n   has been set properly.${ColourReset}")
        unset(HWLOC_LIBRARIES CACHE)
        unset(HWLOC_INCLUDE_DIRS CACHE)
    endif()
endif()

# check a function to validate what was found
if(HWLOC_INCLUDE_DIRS AND HWLOC_LIBRARIES)
    # set required libraries for linking
    ginkgo_set_required_test_lib_link(HWLOC)

    # test linking
    unset(HWLOC_WORKS CACHE)
    include(CheckFunctionExists)
    check_function_exists(hwloc_topology_init HWLOC_WORKS)
    mark_as_advanced(HWLOC_WORKS)

    if(NOT HWLOC_WORKS)
        if(NOT HWLOC_FIND_QUIETLY)
            message(STATUS "Looking for hwloc : test of hwloc_topology_init with hwloc library fails")
            message(STATUS "CMAKE_REQUIRED_LIBRARIES: ${CMAKE_REQUIRED_LIBRARIES}")
            message(STATUS "CMAKE_REQUIRED_INCLUDES: ${CMAKE_REQUIRED_INCLUDES}")
            message(STATUS "CMAKE_REQUIRED_FLAGS: ${CMAKE_REQUIRED_FLAGS}")
            message(STATUS "Check in CMakeFiles/CMakeError.log to figure out why it fails")
        endif()
    endif()
    set(CMAKE_REQUIRED_INCLUDES)
    set(CMAKE_REQUIRED_FLAGS)
    set(CMAKE_REQUIRED_LIBRARIES)

    string(SUBSTRING "${HWLOC_VERSION}" 0 3 HWLOC_VERSION)
    if(HWLOC_VERSION VERSION_LESS HWLOC_FIND_VERSION)
        message(STATUS "Required version ${HWLOC_FIND_VERSION}, but found version ${HWLOC_VERSION}")
        unset(HWLOC_FOUND CACHE)
        unset(HWLOC_LIBRARIES CACHE)
        unset(HWLOC_INCLUDE_DIRS CACHE)
    else()
        unset(HWLOC_FOUND CACHE)
        set(HWLOC_FOUND 1)
        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(HWLOC
            REQUIRED_VARS HWLOC_LIBRARIES HWLOC_INCLUDE_DIRS HWLOC_WORKS
            VERSION_VAR HWLOC_VERSION)
        mark_as_advanced(HWLOC_INCLUDE_DIRS HWLOC_LIBRARIES HWLOC_VERSION HWLOC_WORKS)
    endif()
endif(HWLOC_INCLUDE_DIRS AND HWLOC_LIBRARIES)

if(HWLOC_FOUND AND NOT TARGET hwloc)
    add_library(hwloc SHARED IMPORTED GLOBAL)
    set_target_properties(hwloc PROPERTIES IMPORTED_LOCATION ${HWLOC_LIBRARIES})
    set_target_properties(hwloc PROPERTIES INTERFACE_LINK_LIBRARIES ${HWLOC_LIBRARIES})
    set_target_properties(hwloc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${HWLOC_INCLUDE_DIRS})
endif()

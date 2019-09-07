#.rst:
# FindMETIS
# -------
#
# - Try to find METIS
# Once done this will define
#
#  METIS_FOUND        - system has METIS
#  METIS_INCLUDE_DIRS - include directories for METIS
#  METIS_LIBRARIES    - libraries for METIS
#
# Variables used by this module. They can change the default behaviour and
# need to be set before calling find_package:
#
#  METIS_DIR          - Prefix directory of the METIS installation
#  METIS_INCLUDE_DIR  - Include directory of the METIS installation
#                       (set only if different from ${METIS_DIR}/include)
#  METIS_LIB_DIR      - Library directory of the METIS installation
#                       (set only if different from ${METIS_DIR}/lib)
#  METIS_TEST_RUNS    - Skip tests building and running a test
#                       executable linked against METIS libraries
#  METIS_LIB_SUFFIX   - Also search for non-standard library names with the
#                       given suffix appended
#
# NOTE: This file was modified from a ParMETIS detection script 

#=============================================================================
# Copyright (C) 2015 Jack Poulson. All rights reserved.
#
# Copyright (C) 2010-2012 Garth N. Wells, Anders Logg, Johannes Ring
# and Florian Rathgeber. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

if(NOT METIS_INCLUDE_DIR)
  find_path(METIS_INCLUDE_DIR metis.h
    HINTS ${METIS_INCLUDE_DIR} ENV METIS_INCLUDE_DIR ${METIS_DIR} ENV METIS_DIR
    PATH_SUFFIXES include
    DOC "Directory where the METIS header files are located"
  )
endif()

if(METIS_LIBRARIES)
  set(METIS_LIBRARY ${METIS_LIBRARIES})
endif()
if(NOT METIS_LIBRARY)
  find_library(METIS_LIBRARY
    NAMES metis metis${METIS_LIB_SUFFIX}
    HINTS ${METIS_LIB_DIR} ENV METIS_LIB_DIR ${METIS_DIR} ENV METIS_DIR
    PATH_SUFFIXES lib
    DOC "Directory where the METIS library is located"
  )
endif()

# Get METIS version
if(NOT METIS_VERSION_STRING AND METIS_INCLUDE_DIR AND EXISTS "${METIS_INCLUDE_DIR}/metis.h")
  set(version_pattern "^#define[\t ]+METIS_(MAJOR|MINOR)_VERSION[\t ]+([0-9\\.]+)$")
  file(STRINGS "${METIS_INCLUDE_DIR}/metis.h" metis_version REGEX ${version_pattern})

  foreach(match ${metis_version})
    if(METIS_VERSION_STRING)
      set(METIS_VERSION_STRING "${METIS_VERSION_STRING}.")
    endif()
    string(REGEX REPLACE ${version_pattern} "${METIS_VERSION_STRING}\\2" METIS_VERSION_STRING ${match})
    set(METIS_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  endforeach()
  unset(metis_version)
  unset(version_pattern)
endif()

# Try compiling and running test program
if(METIS_INCLUDE_DIR AND METIS_LIBRARY)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${METIS_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${METIS_LIBRARY})

  # Build and run test program
  include(CheckCSourceRuns)
  check_c_source_runs("
#define METIS_EXPORT
#include \"metis.h\"
int main( int argc, char* argv[] )
{
  // FIXME: Find a simple but sensible test for METIS
  return 0;
}
" METIS_TEST_RUNS)

  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
if(CMAKE_VERSION VERSION_GREATER 2.8.2)
  find_package_handle_standard_args(METIS
    REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR METIS_TEST_RUNS
    VERSION_VAR METIS_VERSION_STRING)
else()
  find_package_handle_standard_args(METIS
    REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR METIS_TEST_RUNS)
endif()

if(METIS_FOUND)
  set(METIS_LIBRARIES ${METIS_LIBRARY})
  set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})

  if(NOT TARGET METIS::METIS)
    add_library(METIS::METIS UNKNOWN IMPORTED)
    set_target_properties(METIS::METIS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIRS}")
    
    if(EXISTS "${METIS_LIBRARIES}")
      set_target_properties(METIS::METIS PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${METIS_LIBRARIES}")
    endif()
    if(METIS_LIBRARY_RELEASE)
      set_property(TARGET METIS::METIS APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(METIS::METIS PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION_RELEASE "${METIS_LIBRARY_RELEASE}")
      unset(METIS_LIBRARY_RELEASE)
    endif()
    if(METIS_LIBRARY_DEBUG)
      set_property(TARGET METIS::METIS APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(METIS::METIS PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION_DEBUG "${METIS_LIBRARY_DEBUG}")
      unset(METIS_LIBRARY_DEBUG)
    endif()
  endif()
endif()

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)

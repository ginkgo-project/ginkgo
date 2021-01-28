###
#
# @copyright (c) 2012-2020 Inria. All rights reserved.
# @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
#
# Copyright 2012-2013 Emmanuel Agullo
# Copyright 2012-2013 Mathieu Faverge
# Copyright 2012      Cedric Castagnede
# Copyright 2013-2020 Florent Pruvost
# Copyright 2020-2021 Ginkgo Project
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

macro(ginkgo_set_required_test_lib_link name)
    set(CMAKE_REQUIRED_INCLUDES "${${name}${STATIC}_INCLUDE_DIRS}")
    if (${name}${STATIC}_CFLAGS_OTHER)
        set(REQUIRED_FLAGS_COPY "${${name}${STATIC}_CFLAGS_OTHER}")
        set(REQUIRED_FLAGS)
        set(REQUIRED_DEFINITIONS)
        foreach(_flag ${REQUIRED_FLAGS_COPY})
            if (_flag MATCHES "^-D")
                list(APPEND REQUIRED_DEFINITIONS "${_flag}")
            endif()
            string(REGEX REPLACE "^-D.*" "" _flag "${_flag}")
            list(APPEND REQUIRED_FLAGS "${_flag}")
        endforeach()
    endif()
    foreach(_var "${REQUIRED_FLAGS_COPY};${REQUIRED_FLAGS};${REQUIRED_LIBRARIES}" )
        if(${_var})
            list(REMOVE_DUPLICATES ${_var})
        endif()
    endforeach()
    set(CMAKE_REQUIRED_DEFINITIONS "${REQUIRED_DEFINITIONS}")
    set(CMAKE_REQUIRED_FLAGS "${REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_LIBRARIES)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${${name}${STATIC}_LDFLAGS_OTHER}")
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${${name}${STATIC}_LIBRARIES}")
    string(REGEX REPLACE "^ -" "-" CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
endmacro()

# Modified function from Morse
macro(ginkgo_check_static_or_dynamic package libraries)
    list(GET ${libraries} 0 _first_lib)
    get_filename_component(_suffix ${_first_lib} EXT)
    if (NOT _suffix)
        unset (_lib_path CACHE)
        find_library(_lib_path ${_first_lib} HINTS ${${package}_LIBDIR} ${${package}_LIBRARY_DIRS} NO_DEFAULT_PATH)
        get_filename_component(_suffix ${_lib_path} EXT)
    endif()
    if (_suffix)
        if(${_suffix} MATCHES ".so$" OR ${_suffix} MATCHES ".dylib$" OR ${_suffix} MATCHES ".dll$")
            set(${package}_STATIC 0)
        elseif(${_suffix} MATCHES ".a$")
            set(${package}_STATIC 1)
        else()
            message(FATAL_ERROR "${package} library extension not in list .a, .so, .dylib, .dll")
        endif()
    else()
        message(FATAL_ERROR "${package} could not detect library extension")
    endif()
endmacro()

function(HEX2DEC str res)
    string(LENGTH "${str}" len)
    if("${len}" EQUAL 1)
        if("${str}" MATCHES "[0-9]")
            set(${res} "${str}" PARENT_SCOPE)
        elseif( "${str}" MATCHES "[aA]")
            set(${res} 10 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[bB]")
            set(${res} 11 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[cC]")
            set(${res} 12 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[dD]")
            set(${res} 13 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[eE]")
            set(${res} 14 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[fF]")
            set(${res} 15 PARENT_SCOPE)
        else()
            return()
        endif()
    else()
        string(SUBSTRING "${str}" 0 1 str1)
        string(SUBSTRING "${str}" 1 -1 str2)
        hex2dec(${str1} res1)
        hex2dec(${str2} res2)
        math(EXPR val "16 * ${res1} + ${res2}")
        set(${res} "${val}" PARENT_SCOPE)
    endif()
endfunction()

macro(get_dec_from_hex hex dec)
    if(${CMAKE_VERSION} VERSION_GREATER 3.13)
        math(EXPR ${dec} ${hex} OUTPUT_FORMAT DECIMAL)
    else()
        hex2dec(${hex} ${dec})
    endif()
endmacro()

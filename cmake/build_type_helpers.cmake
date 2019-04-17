# License:
#
# Copyright (C) 2017 Lectem <lectem@gmail.com>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the 'Software') deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Note: Lectem's original file has gone through several changes to adapt
# to our needs.

include(CMakeDependentOption)

set(${PROJECT_NAME}_CUSTOM_BUILD_TYPES "COVERAGE;TSAN;ASAN" CACHE INTERNAL "")

set(${PROJECT_NAME}_COVERAGE_COMPILER_FLAGS "-g -O0 --coverage" CACHE INTERNAL "")
set(${PROJECT_NAME}_COVERAGE_LINKER_FLAGS   "--coverage"        CACHE INTERNAL "")
set(${PROJECT_NAME}_TSAN_COMPILER_FLAGS "-g -O1 -fsanitize=thread -fno-omit-frame-pointer -fPIC" CACHE INTERNAL "")
set(${PROJECT_NAME}_TSAN_LINKER_FLAGS   "-fsanitize=thread -fno-omit-frame-pointer -fPIC"        CACHE INTERNAL "")
set(${PROJECT_NAME}_ASAN_COMPILER_FLAGS "-g -O1 -fsanitize=address -fno-omit-frame-pointer" CACHE INTERNAL "")
set(${PROJECT_NAME}_ASAN_LINKER_FLAGS   "-fsanitize=address -fno-omit-frame-pointer"        CACHE INTERNAL "")

get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

foreach(_LANG IN LISTS ENABLED_LANGUAGES)
    include(Check${_LANG}CompilerFlag OPTIONAL)
    foreach(_TYPE IN LISTS ${PROJECT_NAME}_CUSTOM_BUILD_TYPES)
        # Required for check_<LANG>_compiler_flag. Caution, this can break several
        # CMake macros, therefore it is __important__ to reset this once we are done.
        set(_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
        set(CMAKE_REQUIRED_LIBRARIES ${${PROJECT_NAME}_${_TYPE}_LINKER_FLAGS})

        if(_LANG STREQUAL "C")
            check_c_compiler_flag("${${PROJECT_NAME}_${_TYPE}_LINKER_FLAGS}"
                ${PROJECT_NAME}_${_LANG}_${_TYPE}_SUPPORTED)
        elseif(_LANG STREQUAL "CXX")
            check_cxx_compiler_flag("${${PROJECT_NAME}_${_TYPE}_LINKER_FLAGS}"
                ${PROJECT_NAME}_${_LANG}_${_TYPE}_SUPPORTED)
        else()
            if(DEFINED ${PROJECT_NAME}_${_LANG}_${_TYPE}_SUPPORTED)
                message(STATUS "Skipping ${_LANG}, not supported by build_type.cmake script")
            endif()
            set(${PROJECT_NAME}_${_LANG}_${_TYPE}_SUPPORTED FALSE)
		        continue()
        endif()
        if(${PROJECT_NAME}_${_LANG}_${_TYPE}_SUPPORTED)
            set(CMAKE_${_LANG}_FLAGS_${_TYPE}
                ${${PROJECT_NAME}_${_TYPE}_COMPILER_FLAGS}
                CACHE STRING "Flags used by the ${_LANG} compiler during ${_TYPE} builds." FORCE
            )
            mark_as_advanced(CMAKE_${_LANG}_FLAGS_${_TYPE})
            set(${PROJECT_NAME}_${_TYPE}_SUPPORTED TRUE CACHE
                STRING "Whether or not coverage is supported by at least one compiler." FORCE)
        endif()
        set(CMAKE_REQUIRED_LIBRARIES ${_CMAKE_REQUIRED_LIBRARIES})
    endforeach()
endforeach()


foreach(_TYPE IN LISTS ${PROJECT_NAME}_CUSTOM_BUILD_TYPES)
    if(${PROJECT_NAME}_${_TYPE}_SUPPORTED)
        set(CMAKE_EXE_LINKER_FLAGS_${_TYPE}
            "${${PROJECT_NAME}_${_TYPE}_LINKER_FLAGS}"
            CACHE STRING "Flags used for linking binaries during ${_TYPE} builds." FORCE
        )
        set(CMAKE_SHARED_LINKER_FLAGS_${_TYPE}
            "${${PROJECT_NAME}_${_TYPE}_LINKER_FLAGS}"
            CACHE STRING "Flags used by the shared libraries linker during ${_TYPE} builds." FORCE
        )
        mark_as_advanced(
           CMAKE_EXE_LINKER_FLAGS_${_TYPE}
           CMAKE_SHARED_LINKER_FLAGS_${_TYPE}
        )
    endif()

    cmake_dependent_option(${PROJECT_NAME}_${_TYPE}_IN_CONFIGURATION_TYPES
        "Should the ${_TYPE} target be in the CMAKE_CONFIGURATION_TYPES list if supported ?" ON
        # No need for this option if we are not using a multi-config generator
        "CMAKE_CONFIGURATION_TYPES;${PROJECT_NAME}_${_TYPE}_SUPPORTED" OFF
    )

    if(${PROJECT_NAME}_${_TYPE}_IN_CONFIGURATION_TYPES)
        # Modify this only if using a multi-config generator
			  # some modules rely on this variable to detect those generators.
        if(CMAKE_CONFIGURATION_TYPES AND ${PROJECT_NAME}_${_TYPE}_SUPPORTED)
            list(APPEND CMAKE_CONFIGURATION_TYPES ${_TYPE})
            list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
            set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
                "Semicolon separated list of supported configuration types, only supports ${CMAKE_CONFIGURATION_TYPES} anything else will be ignored."
                FORCE
            )
        endif()
    else()
        if(${_TYPE} IN_LIST CMAKE_CONFIGURATION_TYPES)
            message(STATUS "Removing ${_TYPE} configuration type (${PROJECT_NAME}_${_TYPE}_IN_CONFIGURATION_TYPES is OFF)")
            list(REMOVE_ITEM CMAKE_CONFIGURATION_TYPES ${_TYPE})
            list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
            set(CMAKE_CONFIGURATION_TYPES
                "${CMAKE_CONFIGURATION_TYPES}"
                CACHE STRING
                "Semicolon separated list of supported configuration types, only supports ${CMAKE_CONFIGURATION_TYPES} anything else will be ignored."
                FORCE
            )
        endif()
    endif()
endforeach()

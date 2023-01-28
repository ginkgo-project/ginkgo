# TODO: we may use file(GENERATE ... TARGET ...) to generate config file based on the target property
#       when we bump the CMake minimum version to 3.15/3.19
function(filter_generator_expressions INPUT OUTPUT)
    # See https://gitlab.kitware.com/cmake/cmake/-/blob/v3.22.2/Modules/FindMPI.cmake#L1218
    # and other versions of this file for what we are removing here.
    string(REGEX REPLACE "[$<]+COMPILE.*>:.+[>]+(.+)" "\\1" TMP "${INPUT}")
    # There can be at least two type of SHELL, one with generator $<> form, one
    # without (v3.16.x). Sometimes, it also has extra arguments. We need to do
    # at least a greedy regex also consuming the final `>` if present before
    # doing a non greedy one for the leftovers.
    string(REGEX REPLACE "[$<A-Z_:]+SHELL:(.+)>+" "\\1" TMP "${TMP}")
    string(REGEX REPLACE "[$<A-Z_:]*SHELL:(.+)>*" "\\1" TMP "${TMP}")
    string(REGEX REPLACE ".+INTERFACE:.+>" "" TMP "${TMP}")
    string(REGEX REPLACE "\$<COMMA>" "," TMP "${TMP}")
    # Remove the left : or > 
    string(REGEX REPLACE ":|>" "" TMP "${TMP}")
    # Ignore hwloc include if it is the internal one
    string(REGEX REPLACE "${PROJECT_BINARY_DIR}.*hwloc/src/include.*" "" TMP "${TMP}")
    set(${OUTPUT} "${TMP}" PARENT_SCOPE)
endfunction()

macro(ginkgo_interface_libraries_recursively INTERFACE_LIBS)
    # always add the interface to the list to keep the order information 
    # Currently, it does not support the circular dependence and MSVC.
    foreach(_libs ${INTERFACE_LIBS})
        if (TARGET ${_libs})
            if("${_libs}" MATCHES "ginkgo.*")
                set(GINKGO_INTERFACE_LIB_NAME "-l${_libs}$<$<CONFIG:Debug>:${CMAKE_DEBUG_POSTFIX}>")
                list(APPEND GINKGO_INTERFACE_LIBS_FOUND "${GINKGO_INTERFACE_LIB_NAME}")
            endif()
            # Get the link flags and treat them
            get_target_property(GINKGO_INTERFACE_LIBS_LINK_FLAGS "${_libs}"
                INTERFACE_LINK_OPTIONS)
            if (GINKGO_INTERFACE_LIBS_LINK_FLAGS)
                filter_generator_expressions("${GINKGO_INTERFACE_LIBS_LINK_FLAGS}"
                    GINKGO_INTERFACE_LIB_NAME)
            endif()
            # Get the imported library
            get_target_property(_libs_type "${_libs}" TYPE)
            get_target_property(_libs_imported "${_libs}" IMPORTED)
            if (_libs_imported AND NOT ${_libs_type} STREQUAL "INTERFACE_LIBRARY")
                get_target_property(GINKGO_LIBS_IMPORTED_LIBS "${_libs}" IMPORTED_LOCATION_RELEASE)
                if (NOT GINKGO_LIBS_IMPORTED_LIBS)
                    get_target_property(GINKGO_LIBS_IMPORTED_LIBS "${_libs}" IMPORTED_LOCATION)
                endif()
                if (GINKGO_LIBS_IMPORTED_LIBS)
                    list(APPEND GINKGO_INTERFACE_LIBS_FOUND "${GINKGO_LIBS_IMPORTED_LIBS}")
                endif()
            endif()
            # Populate the include directories
            get_target_property(GINKGO_LIBS_INTERFACE_INCS "${_libs}"
                INTERFACE_INCLUDE_DIRECTORIES)
            foreach(_incs ${GINKGO_LIBS_INTERFACE_INCS})
                filter_generator_expressions("${_incs}" GINKGO_INTERFACE_INC_FILTERED)
                if (GINKGO_INTERFACE_INC_FILTERED AND NOT
                        "-I${GINKGO_INTERFACE_INC_FILTERED}" IN_LIST GINKGO_INTERFACE_CFLAGS_FOUND)
                    list(APPEND GINKGO_INTERFACE_CFLAGS_FOUND "-I${GINKGO_INTERFACE_INC_FILTERED}")
                endif()
            endforeach()

            # Populate the compiler options and definitions if needed
            get_target_property(GINKGO_LIBS_INTERFACE_DEFS "${_libs}"
                INTERFACE_COMPILE_DEFINITIONS)
            if (GINKGO_LIBS_INTERFACE_DEFS)
                list(APPEND GINKGO_INTERFACE_CFLAGS_FOUND "${GINKGO_LIBS_INTERFACE_DEFS}")
            endif()
            get_target_property(GINKGO_LIBS_INTERFACE_OPTS "${_libs}"
                INTERFACE_COMPILE_OPTIONS)
            filter_generator_expressions("${GINKGO_LIBS_INTERFACE_OPTS}" GINKGO_LIBS_INTERFACE_OPTS_FILTERED)
            if (GINKGO_LIBS_INTERFACE_OPTS)
                list(APPEND GINKGO_INTERFACE_CFLAGS_FOUND "${GINKGO_LIBS_INTERFACE_OPTS_FILTERED}")
            endif()

            # Keep recursing through the libraries
            get_target_property(GINKGO_LIBS_INTERFACE_LIBS "${_libs}"
                INTERFACE_LINK_LIBRARIES)
            # removing $<LINK_ONLY:>
            list(TRANSFORM GINKGO_LIBS_INTERFACE_LIBS REPLACE "\\$<LINK_ONLY:(.*)>" "\\1")
            ginkgo_interface_libraries_recursively("${GINKGO_LIBS_INTERFACE_LIBS}")
        elseif(EXISTS "${_libs}")
            if ("${_libs}" MATCHES "${PROJECT_BINARY_DIR}.*hwloc.so")
                list(APPEND GINKGO_INTERFACE_LIBS_FOUND "${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_LIBRARY_DIR}/libhwloc.so")
            else()
                list(APPEND GINKGO_INTERFACE_LIBS_FOUND "${_libs}")
            endif()
        elseif("${_libs}" STREQUAL "${CMAKE_DL_LIBS}")
            list(APPEND GINKGO_INTERFACE_LIBS_FOUND "-l${_libs}")
        endif()
    endforeach()
endmacro()

macro(ginkgo_interface_information)
    set(GINKGO_INTERFACE_LINK_FLAGS "-L${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_LIBRARY_DIR}")
    unset(GINKGO_INTERFACE_LIBS_FOUND)
    unset(GINKGO_INTERFACE_CFLAGS_FOUND)
    # Prepare recursively populated library list
    list(APPEND GINKGO_INTERFACE_LIBS_FOUND "-lginkgo$<$<CONFIG:Debug>:${CMAKE_DEBUG_POSTFIX}>")
    # Prepare recursively populated include directory list
    list(APPEND GINKGO_INTERFACE_CFLAGS_FOUND
        "-I${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_INCLUDE_DIR}")

    # Call the recursive interface libraries macro
    get_target_property(GINKGO_INTERFACE_LINK_LIBRARIES ginkgo INTERFACE_LINK_LIBRARIES)
    ginkgo_interface_libraries_recursively("${GINKGO_INTERFACE_LINK_LIBRARIES}")
    # Format and store the interface libraries found
    # remove duplicates on the reversed list to keep the dependecy in the end of list.
    list(REVERSE GINKGO_INTERFACE_LIBS_FOUND)
    list(REMOVE_DUPLICATES GINKGO_INTERFACE_LIBS_FOUND)
    list(REVERSE GINKGO_INTERFACE_LIBS_FOUND)
    list(REMOVE_ITEM GINKGO_INTERFACE_LIBS_FOUND "")
    # keep it as list 
    set(GINKGO_INTERFACE_LINK_FLAGS
        ${GINKGO_INTERFACE_LINK_FLAGS} ${GINKGO_INTERFACE_LIBS_FOUND})
    unset(GINKGO_INTERFACE_LIBS_FOUND)
    # Format and store the interface cflags found
    list(REMOVE_DUPLICATES GINKGO_INTERFACE_CFLAGS_FOUND)
    list(REMOVE_ITEM GINKGO_INTERFACE_CFLAGS_FOUND "")
    # Keep it as list
    set(GINKGO_INTERFACE_CXX_FLAGS ${GINKGO_INTERFACE_CFLAGS_FOUND})
    unset(GINKGO_INTERFACE_CFLAGS_FOUND)
endmacro(ginkgo_interface_information)

macro(ginkgo_git_information)
    if(EXISTS "${Ginkgo_SOURCE_DIR}/.git")
        find_package(Git QUIET)
        if(GIT_FOUND)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} describe --contains --all HEAD
                WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
                OUTPUT_VARIABLE GINKGO_GIT_BRANCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%H ${Ginkgo_SOURCE_DIR}
                WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
                OUTPUT_VARIABLE GINKGO_GIT_REVISION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%h ${Ginkgo_SOURCE_DIR}
                WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
                OUTPUT_VARIABLE GINKGO_GIT_SHORTREV
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
    endif()
endmacro(ginkgo_git_information)

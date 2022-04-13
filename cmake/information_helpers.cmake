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
    # Ignore hwloc include if it is the internal one
    string(REGEX REPLACE "${PROJECT_BINARY_DIR}.*hwloc/src/include.*" "" TMP "${TMP}")
    set(${OUTPUT} "${TMP}" PARENT_SCOPE)
endfunction()

macro(ginkgo_interface_libraries_recursively INTERFACE_LIBS)
    foreach(_libs ${INTERFACE_LIBS})
        if (NOT "${_libs}" IN_LIST GINKGO_INTERFACE_LIBS_FOUND
                AND NOT "-l${_libs}" IN_LIST GINKGO_INTERFACE_LIBS_FOUND)
            if (TARGET ${_libs})
                if (upper_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND "${_libs}" MATCHES "ginkgo.*")
                    set(GINKGO_INTERFACE_LIB_NAME "-l${_libs}${CMAKE_DEBUG_POSTFIX}")
                elseif("${_libs}" MATCHES "ginkgo.*") # Ginkgo libs are appended in the form -l
                    set(GINKGO_INTERFACE_LIB_NAME "-l${_libs}")
                endif()

                # Get the link flags and treat them
                get_target_property(GINKGO_INTERFACE_LIBS_LINK_FLAGS "${_libs}"
                    INTERFACE_LINK_OPTIONS)
                if (GINKGO_INTERFACE_LIBS_LINK_FLAGS)
                    filter_generator_expressions("${GINKGO_INTERFACE_LIBS_LINK_FLAGS}"
                        GINKGO_INTERFACE_LIB_NAME)
                endif()
                if (NOT "${GINKGO_INTERFACE_LIB_NAME}" IN_LIST GINKGO_INTERFACE_LIBS_FOUND)
                    list(APPEND GINKGO_INTERFACE_LIBS_FOUND "${GINKGO_INTERFACE_LIB_NAME}")
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
                ginkgo_interface_libraries_recursively("${GINKGO_LIBS_INTERFACE_LIBS}")
            elseif(EXISTS "${_libs}")
                if ("${_libs}" MATCHES "${PROJECT_BINARY_DIR}.*hwloc.so")
                    list(APPEND GINKGO_INTERFACE_LIBS_FOUND "${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_LIBRARY_DIR}/libhwloc.so")
                else()
                    list(APPEND GINKGO_INTERFACE_LIBS_FOUND "${_libs}")
                endif()
            endif()
        endif()
    endforeach()
endmacro()

macro(ginkgo_interface_information)
    set(GINKGO_INTERFACE_LINK_FLAGS "-L${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_LIBRARY_DIR}")
    unset(GINKGO_INTERFACE_LIBS_FOUND)
    unset(GINKGO_INTERFACE_CFLAGS_FOUND)
    # Prepare recursively populated library list
    string(TOUPPER "${CMAKE_BUILD_TYPE}" upper_CMAKE_BUILD_TYPE)
    if (upper_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        list(APPEND GINKGO_INTERFACE_LIBS_FOUND "-lginkgo${CMAKE_DEBUG_POSTFIX}")
    else()
        list(APPEND GINKGO_INTERFACE_LIBS_FOUND "-lginkgo")
    endif()
    # Prepare recursively populated include directory list
    list(APPEND GINKGO_INTERFACE_CFLAGS_FOUND
        "-I${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_INCLUDE_DIR}")

    # Call the recursive interface libraries macro
    get_target_property(GINKGO_INTERFACE_LINK_LIBRARIES ginkgo INTERFACE_LINK_LIBRARIES)
    ginkgo_interface_libraries_recursively("${GINKGO_INTERFACE_LINK_LIBRARIES}")

    # Format and store the interface libraries found
    list(REMOVE_DUPLICATES GINKGO_INTERFACE_LIBS_FOUND)
    list(REMOVE_ITEM GINKGO_INTERFACE_LIBS_FOUND "")
    string(REPLACE ";" " "
        GINKGO_FORMATTED_INTERFACE_LIBS_FOUND "${GINKGO_INTERFACE_LIBS_FOUND}")
    set(GINKGO_INTERFACE_LINK_FLAGS
        "${GINKGO_INTERFACE_LINK_FLAGS} ${GINKGO_FORMATTED_INTERFACE_LIBS_FOUND}")
    unset(GINKGO_INTERFACE_LIBS_FOUND)
    # Format and store the interface cflags found
    list(REMOVE_DUPLICATES GINKGO_INTERFACE_CFLAGS_FOUND)
    list(REMOVE_ITEM GINKGO_INTERFACE_CFLAGS_FOUND "")
    string(REPLACE ";" " "
        GINKGO_FORMATTED_INTERFACE_CFLAGS_FOUND "${GINKGO_INTERFACE_CFLAGS_FOUND}")
    set(GINKGO_INTERFACE_CXX_FLAGS "${GINKGO_FORMATTED_INTERFACE_CFLAGS_FOUND}")
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

macro(ginkgo_interface_libraries_recursively INTERFACE_LIBS)
    foreach(_libs ${INTERFACE_LIBS})
        if (TARGET ${_libs})
            if (upper_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND "${_libs}" MATCHES "ginkgo*")
                list(APPEND GINKGO_INTERFACE_LIBS_FOUND "-l${_libs}${CMAKE_DEBUG_POSTFIX}")
            else()
                list(APPEND GINKGO_INTERFACE_LIBS_FOUND "-l${_libs}")
            endif()
            get_target_property(GINKGO_LIBS_INTERFACE_LIBS "${_libs}" INTERFACE_LINK_LIBRARIES)
            ginkgo_interface_libraries_recursively("${GINKGO_LIBS_INTERFACE_LIBS}")
        endif()
    endforeach()
endmacro()

macro(ginkgo_interface_information)
    get_target_property(GINKGO_INTERFACE_LINK_LIBRARIES ginkgo INTERFACE_LINK_LIBRARIES)
    set(GINKGO_INTERFACE_LINK_FLAGS "-L${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_LIBRARY_DIR}")
    set(GINKGO_INTERFACE_CXX_FLAGS "-I${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_INCLUDE_DIR}")

    unset(GINKGO_INTERFACE_LIBS_FOUND)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" upper_CMAKE_BUILD_TYPE)
    if (upper_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        list(APPEND GINKGO_INTERFACE_LIBS_FOUND "-lginkgo${CMAKE_DEBUG_POSTFIX}")
    else()
        list(APPEND GINKGO_INTERFACE_LIBS_FOUND "-lginkgo")
    endif()
    ginkgo_interface_libraries_recursively("${GINKGO_INTERFACE_LINK_LIBRARIES}")
    list(REMOVE_DUPLICATES GINKGO_INTERFACE_LIBS_FOUND)
    string(REPLACE ";" " " GINKGO_FORMATTED_INTERFACE_LIBS_FOUND "${GINKGO_INTERFACE_LIBS_FOUND}")
    set(GINKGO_INTERFACE_LINK_FLAGS "${GINKGO_INTERFACE_LINK_FLAGS} ${GINKGO_FORMATTED_INTERFACE_LIBS_FOUND}")
    unset(GINKGO_INTERFACE_LIBS_FOUND)
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

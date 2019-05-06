macro(ginkgo_interface_information)
    get_target_property(GINKGO_INTERFACE_LINK_LIBRARIES ginkgo INTERFACE_LINK_LIBRARIES)
		set(GINKGO_INTERFACE_LINK_FLAGS "-L${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_LIBRARY_DIR} -lginkgo")
		set(GINKGO_INTERFACE_CXX_FLAGS "-I${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_INCLUDE_DIR}")

		foreach(_libs IN LISTS GINKGO_INTERFACE_LINK_LIBRARIES)
			  set(GINKGO_INTERFACE_LINK_FLAGS "${GINKGO_INTERFACE_LINK_FLAGS} -l${_libs}")
		endforeach()
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
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%H ${Gingko_SOURCE_DIR}
                WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
                OUTPUT_VARIABLE GINKGO_GIT_REVISION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%h ${Gingko_SOURCE_DIR}
                WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
                OUTPUT_VARIABLE GINKGO_GIT_SHORTREV
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
    endif()
endmacro(ginkgo_git_information)

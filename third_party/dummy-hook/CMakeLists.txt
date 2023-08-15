# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --git-path hooks
      WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
      OUTPUT_VARIABLE Ginkgo_GIT_HOOKS_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    # If it is not in git, give an error and return
    if(NOT Ginkgo_GIT_HOOKS_DIR)
        message(STATUS "Ginkgo is not in a git repository, so no git hook was installed")
        return()
    endif()
    get_filename_component(Ginkgo_GIT_HOOKS_DIR ${Ginkgo_GIT_HOOKS_DIR} REALPATH BASE_DIR ${Ginkgo_SOURCE_DIR})
    if(EXISTS "${Ginkgo_GIT_HOOKS_DIR}")
        set(ADD_HOOK FALSE)
        set(HOOK_LOCATION "${Ginkgo_GIT_HOOKS_DIR}/pre-commit")
        if(NOT EXISTS "${HOOK_LOCATION}")
            set(ADD_HOOK TRUE)
        else()
            # check if the correct hook is installed
            execute_process(COMMAND grep git-cmake-format.py "${HOOK_LOCATION}"
                            RESULT_VARIABLE res OUTPUT_QUIET)
            # return value =/= 0 means the pattern was not found
            if(NOT res EQUAL 0)
                set(ADD_HOOK TRUE)
            endif()
        endif()
        if(ADD_HOOK)
            configure_file(dummy_hook "${HOOK_LOCATION}" COPYONLY)
        endif()
    endif()
endif()

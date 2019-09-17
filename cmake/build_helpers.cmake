function(ginkgo_default_includes name)
    # set include path depending on used interface
    target_include_directories("${name}"
        PUBLIC
            $<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        )
endfunction()

function(ginkgo_compile_features name)
    target_compile_features("${name}" PUBLIC cxx_std_11)
    if(GINKGO_WITH_CLANG_TIDY AND GINKGO_CLANG_TIDY_PATH)
        set_property(TARGET "${name}" PROPERTY CXX_CLANG_TIDY "${GINKGO_CLANG_TIDY_PATH};-checks=*")
    endif()
    if(GINKGO_WITH_IWYU AND GINKGO_IWYU_PATH)
        set_property(TARGET "${name}" PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${GINKGO_IWYU_PATH})
    endif()
endfunction()

macro(ginkgo_prepare_windows_path)
    list(APPEND GINKGO_WINDOWS_PATH ${CMAKE_CURRENT_BINARY_DIR})
    set(GINKGO_WINDOWS_PATH ${GINKGO_WINDOWS_PATH} PARENT_SCOPE)
endmacro()

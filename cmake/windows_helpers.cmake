function(ginkgo_switch_windows_link lang from to)
    foreach(flag_var
        "CMAKE_${lang}_FLAGS" "CMAKE_${lang}_FLAGS_DEBUG" "CMAKE_${lang}_FLAGS_RELEASE"
        "CMAKE_${lang}_FLAGS_MINSIZEREL" "CMAKE_${lang}_FLAGS_RELWITHDEBINFO"
        )
        if(${flag_var} MATCHES "/${from}")
            string(REGEX REPLACE "/${from}" "/${to}" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/${from}")
        if(${flag_var} MATCHES "-${from}")
            string(REGEX REPLACE "-${from}" "-${to}" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "-${from}")
        set(${flag_var} "${${flag_var}}" CACHE STRING "" FORCE)
    endforeach()
endfunction()

macro(ginkgo_switch_to_windows_static lang)
    ginkgo_switch_windows_link(${lang} "MD" "MT")
endmacro()

macro(ginkgo_switch_to_windows_dynamic lang)
    ginkgo_switch_windows_link(${lang} "MT" "MD")
endmacro()

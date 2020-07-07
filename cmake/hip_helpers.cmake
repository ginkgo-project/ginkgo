macro(ginkgo_hip_ban_link_hcflag target)
    if(TARGET ${target})
        get_target_property(GINKGO_TARGET_ILL ${target} INTERFACE_LINK_LIBRARIES)
        string(REPLACE "-hc " "" GINKGO_TARGET_NEW_ILL "${GINKGO_TARGET_ILL}")
        set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES "${GINKGO_TARGET_NEW_ILL}")
    endif()
endmacro()

macro(ginkgo_hip_ban_compile_hcflag target)
    if(TARGET ${target})
        get_target_property(GINKGO_TARGET_ILL ${target} INTERFACE_COMPILE_OPTIONS)
        string(REPLACE "-hc" "" GINKGO_TARGET_NEW_ILL "${GINKGO_TARGET_ILL}")
        set_target_properties(${target} PROPERTIES INTERFACE_COMPILE_OPTIONS "${GINKGO_TARGET_NEW_ILL}")
    endif()
endmacro()

macro(ginkgo_hip_clang_ban_hip_device_flags)
    if (GINKGO_HIP_VERSION VERSION_GREATER_EQUAL "3.5")
        # Compile options somehow add hip-clang specific flags. Wipe them.
        # Currently, the flags wiped out should be:
        # -x;hip;--hip-device-lib-path=/opt/rocm/lib;--cuda-gpu-arch=gfx900;
        # --cuda-gpu-arch=gfx906
        set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "")
        # In addition, link libraries have a similar problem. We only keep
        # `hip::host`. Currently, the flags should be:
        # hip::host;--hip-device-lib-path=/opt/rocm/lib;--hip-link;
        # --cuda-gpu-arch=gfx900;--cuda-gpu-arch=gfx906
        set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")
    endif()
endmacro()

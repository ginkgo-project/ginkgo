# Resolves GINKGO_OMP_SIMD_ARCH into gko_simd_flags, then probes which Highway
# target that actually produces. -march=haswell and -march=x86-64-v3 resolve to
# SSSE3 unless HWY_DISABLE_PCLMUL_AES drops Highway's __AES__ requirement; the
# probe confirms it worked.
function(gko_resolve_simd_flags out_var)
    if(GINKGO_OMP_SIMD_ARCH STREQUAL "native")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
            set(${out_var} "-mcpu=native" PARENT_SCOPE)
        else()
            set(${out_var} "-march=native" PARENT_SCOPE)
        endif()
    elseif(GINKGO_OMP_SIMD_ARCH STREQUAL "baseline")
        set(${out_var} "" PARENT_SCOPE)
    else()
        separate_arguments(_flags NATIVE_COMMAND "${GINKGO_OMP_SIMD_ARCH}")
        set(${out_var} "${_flags}" PARENT_SCOPE)
    endif()
endfunction()

# Applies the complete Highway wiring to one target: the library, the compile
# definitions the layer relies on, and the resolved ISA flags. Every target that
# compiles omp/base/kernel_launch_simd.hpp needs all three, and they must agree.
#
# HWY_COMPILE_ONLY_STATIC: dynamic dispatch needs an HWY_EXPORT entry per
#   concrete function; our launcher is a header template with no such set.
# HWY_DISABLE_ATTR: otherwise every kernel lambda needs HWY_ATTR before its
#   opening brace, which GKO_KERNEL's position makes inexpressible.
# HWY_DISABLE_PCLMUL_AES: Ginkgo uses no AES/CLMUL ops; requiring them
#   downgrades -march=haswell and -march=x86-64-vN to SSSE3.
function(gko_add_highway target)
    if(NOT TARGET hwy::hwy)
        message(
            FATAL_ERROR
            "Ginkgo OMP SIMD: GINKGO_OMP_WITH_HIGHWAY is ON but the "
            "`hwy::hwy` target does not exist. Neither find_package(HWY) nor "
            "third_party/highway provided it."
        )
    endif()
    target_link_libraries(${target} PRIVATE hwy::hwy)
    target_compile_definitions(
        ${target}
        PRIVATE
            GKO_OMP_HIGHWAY
            HWY_COMPILE_ONLY_STATIC
            HWY_DISABLE_ATTR
            HWY_DISABLE_PCLMUL_AES
    )
    gko_resolve_simd_flags(_gko_simd_flags)
    if(_gko_simd_flags)
        # Applied to the whole target, never per source file: compiling header
        # templates at differing ISA levels across TUs risks the linker
        # selecting an AVX-512 instantiation of an inline function
        # program-wide.
        target_compile_options(${target} PRIVATE ${_gko_simd_flags})
    endif()
endfunction()

function(gko_probe_highway simd_flags hwy_include_dir)
    set(_defs
        -DHWY_COMPILE_ONLY_STATIC
        -DHWY_DISABLE_ATTR
        -DHWY_DISABLE_PCLMUL_AES
    )
    # Prefixed because try_run writes its result variables into CMakeCache.txt
    # verbatim, where an unprefixed `_compile_result` could collide with a
    # project embedding Ginkgo.
    try_run(
        GKO_HWY_PROBE_RUN_RESULT
        GKO_HWY_PROBE_COMPILE_RESULT
        "${CMAKE_BINARY_DIR}/CMakeFiles/hwy_probe"
        "${PROJECT_SOURCE_DIR}/cmake/hwy_probe.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${hwy_include_dir}"
        COMPILE_DEFINITIONS ${_defs} ${simd_flags}
        RUN_OUTPUT_VARIABLE _probe_out
        COMPILE_OUTPUT_VARIABLE _probe_compile_out
    )
    if(NOT GKO_HWY_PROBE_COMPILE_RESULT)
        # GINKGO_OMP_WITH_HIGHWAY is ON and the probe is the same header, the
        # same definitions and the same ISA flags the layer itself is built
        # with. If that does not compile, neither will ginkgo_omp, so failing
        # here is strictly better than failing later inside a kernel.
        message(
            FATAL_ERROR
            "Ginkgo OMP SIMD: the Highway probe failed to compile with "
            "GINKGO_OMP_SIMD_ARCH='${GINKGO_OMP_SIMD_ARCH}'. Either fix the "
            "flags or configure with -DGINKGO_OMP_WITH_HIGHWAY=OFF. "
            "Output:\n${_probe_compile_out}"
        )
    endif()
    if(NOT GKO_HWY_PROBE_RUN_RESULT EQUAL 0)
        # Cross-compiling: the probe cannot execute. Compile succeeded, so the
        # configuration is usable; we simply cannot report the lane count.
        message(
            STATUS
            "Ginkgo OMP SIMD: highway 1.4.0 (target not probed, cross-compiling)"
        )
        set(GKO_HWY_TARGET_NAME "cross" PARENT_SCOPE)
        set(GKO_HWY_F64_LANES "0" PARENT_SCOPE)
        return()
    endif()
    string(STRIP "${_probe_out}" _probe_out)
    string(REGEX REPLACE "^N_" "" _name "${_probe_out}")
    string(REGEX REPLACE " .*$" "" _name "${_name}")
    string(REGEX REPLACE "^.* " "" _lanes "${_probe_out}")
    set(GKO_HWY_TARGET_NAME "${_name}" PARENT_SCOPE)
    set(GKO_HWY_F64_LANES "${_lanes}" PARENT_SCOPE)
    message(
        STATUS
        "Ginkgo OMP SIMD: highway 1.4.0, target ${_name}, ${_lanes} x f64"
    )
    if(NOT GINKGO_OMP_SIMD_ARCH STREQUAL "baseline" AND _lanes LESS_EQUAL 2)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|X86|amd64|AMD64|i.86")
            # The named-CPU advice below is x86-specific: on AArch64 without
            # SVE, 2 x f64 is the NEON register width and there is nothing to
            # fix, so suggesting AMD/Intel CPU names there would be nonsense.
            message(
                WARNING
                "Ginkgo OMP SIMD: GINKGO_OMP_SIMD_ARCH='${GINKGO_OMP_SIMD_ARCH}' "
                "resolved to only ${_lanes} double lanes (target ${_name}). "
                "On x86, -march=haswell and -march=x86-64-vN do not enable AES "
                "and therefore do not reach Highway's AVX2 baseline. Use "
                "-DGINKGO_OMP_SIMD_ARCH=native, or a named CPU such as "
                "-march=znver4 / -march=skylake-avx512."
            )
        else()
            message(
                STATUS
                "Ginkgo OMP SIMD: ${_lanes} double lanes (target ${_name}) on "
                "${CMAKE_SYSTEM_PROCESSOR}. If this machine has a wider vector "
                "unit (SVE, RVV), pass its flags via GINKGO_OMP_SIMD_ARCH."
            )
        endif()
    endif()
endfunction()

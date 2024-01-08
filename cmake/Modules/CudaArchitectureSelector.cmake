# SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

#.rst:
# CudaArchitectureSelector
# ------------------------
#
# Sets the target architectures for which to compile CUDA code.
#
# Functions
# ^^^^^^^^^
#
# CudaArchitectureSelector exposes the following function:
#
# ::
# 
#   cas_target_cuda_architectures(
#    <target>                   # target for which to set the architectures
#    [ARCHITECTURES <spec>...]  # list of architecture specifications
#    [UNSUPPORTED <arch>...]    # list of architectures not supported by the
#                               # target
#   )
#
# The command adds the appropriate list of additional compiler flags so that the
# CUDA sources of the target are compiled for all architectures described in the
# specification. Optionally, the list of flags can be filtered by specifying
# which architectures are not supported by the target.
#
# ::
# 
#   cas_variable_cuda_architectures(
#    <variable>                 # variable for storing architectures compiler 
#                               # flag
#    [ARCHITECTURES <spec>...]  # list of architecture specifications
#    [UNSUPPORTED <arch>...]    # list of architectures not supported
#   )
#
# The command has the same result as ``cas_target_cuda_architectures``. It does 
# not add the compiler flags to the target, but stores the compiler flags in 
# the variable (string).
# 
#   cas_variable_cmake_cuda_architectures(
#    [<variable>]               # variable for storing architecture list
#    [<spec>]                   # list of architecture specifications
#   )
#
# The command prepares an architecture list supported by the CMake
# ``CUDA_ARCHITECTURES`` target property and ``CMAKE_CUDA_ARCHITECTURES``
# variable. The architecture specification 
#
# 
# ``ARCHITECTURES`` specification list
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The specification list passed as the ``ARCHITECTURES`` parameter can contain
# the following types of entries (entries of distinct types can be combined in
# the same list):
#
# Automatic detection:
#   Specified by the string ``Auto``. Flags for CUBIN code generation will be
#   added for all GPU architectures detected on the system. When nothing is
#   detected, the behavior is the same as ``All``.
#
# All available architectures:
#   Specified by the string ``All``. This option will add flags for CUBIN code
#   generation for all GPU architectures supported by the compiler.
#
# GPU generation name:
#   Has to be one of the strings ``Tesla``, ``Fermi``, ``Kepler``, ``Maxwell``,
#   ``Pascal``, ``Volta``, ``Turing``, ``Ampere``. Specifying one of the strings
#   will add flags for the generation of CUBIN code for all architectures
#   belonging to that GPU generation (except the ones listed in the
#   ``UNSUPPORTED`` list).
#
# Virtual and physical architecture specification:
#   A string of the form ``XX(YY)``, where ``XX`` is the identifier of the
#   physical architecture (e.g. ``XX=32`` represent the physical architecture
#   ``sm_32``) and ``YY`` is the identifier of the virtual architecture (e.g.
#   ``YY=52`` represents the virtual architecture ``compute_52``).
#   Flags necessary to generate CUBIN code for the specified combination of the
#   physical and virtual architecture will be added to the compiler flags.
#
# Only physical architecture specification
#   A string of the form ``XX``. Functionally exactly equivalent to ``XX(XX)``.
#
# Only virtual architecture specification
#   A string of the form ``(YY)``, where ``YY`` is the identifier of the virtual
#   architecture. Flags necessary to generate the PTX code (which can be used by
#   the jitter to compile for the specific device at runtime) for the specified
#   architecture will be added to the compiler flags.
#
# clang-cuda makes no distinction between virtual and physical architecture,
# and the physical architecture takes precedence over the virtual architecture.
#
# ``UNSUPPORTED`` architectures list
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This list has entries of the form ``XX``, which represent the architecture
# (both physical and virtual) identifiers. All code generation for both the
# physical (CUBIN) and virtual (PTX) architectures matching one of the
# identifiers in this list will be removed from the list specified by the
# ``ARCHITECTURES`` list. A warning will be printed for each removed entry.
# The list also supports aggregates ``All``, ``Auto`` and GPU generation names
# which have the same meaning as in the ``ARCHITECTURES'' specification list.


if(NOT DEFINED CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA language support is not enabled")
endif()


set(cas_spec_regex "^([0-9]+)?(\\(([0-9]+)\\))?\$")


# returns a list of GPU architectures supported by the compiler
function(cas_get_supported_architectures output)
    if(DEFINED CAS_SUPPORTED_ARCHITECTURES)
        set(${output} ${CAS_SUPPORTED_ARCHITECTURES} PARENT_SCOPE)
        return()
    endif()
    set(CAS_NVCC_EXEC "${CMAKE_CUDA_COMPILER}")
    # for clang-cuda determine underlying nvcc location
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
        execute_process(
            COMMAND "${CMAKE_CUDA_COMPILER}" -v
            RESULT_VARIABLE status
            ERROR_VARIABLE clang_info_content
            OUTPUT_QUIET)
        if(NOT (status EQUAL 0))
            message(FATAL_ERROR "Unable to execute clang-cuda")
        endif()
        if (clang_info_content MATCHES "Found CUDA installation: (.*), version ")
            set(CAS_NVCC_EXEC "${CMAKE_MATCH_1}/bin/nvcc")
        else()
            message(FATAL_ERROR "Unable to determine CUDA installation path from clang-cuda")
        endif()
    endif()
    execute_process(
        COMMAND "${CAS_NVCC_EXEC}" --help
        RESULT_VARIABLE status
        OUTPUT_VARIABLE help_content
        ERROR_QUIET)
    if(NOT (status EQUAL 0))
        message(FATAL_ERROR "Unable to determine supported GPU architectures")
    endif()
    string(REGEX MATCHALL "sm_[0-9]+" extracted_info ${help_content})
    list(SORT extracted_info)
    list(REMOVE_DUPLICATES extracted_info)
    foreach(item IN LISTS extracted_info)
        set(detector_name "${PROJECT_BINARY_DIR}/CMakeFiles/cas_supported.cu")
        file(WRITE "${detector_name}"
            "int main() {"
            "  return 0;"
            "}")
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CUDA_FLAGS "--cuda-gpu-arch=${item}")
        else()
            set(CMAKE_CUDA_FLAGS "--gpu-architecture=${item}")
        endif()
        try_compile(status "${PROJECT_BINARY_DIR}"
            SOURCES "${detector_name}")
        if (status)
            string(REGEX REPLACE "sm_([0-9]+)" "\\1" temp ${item})
            list(APPEND supported ${temp})
        endif()
        file(REMOVE "${detector_name}")
        unset(CMAKE_CUDA_FLAGS)
    endforeach()
    if(NOT DEFINED supported)
        message(FATAL_ERROR "Unable to determine supported GPU architectures")
    endif()
    set(CAS_SUPPORTED_ARCHITECTURES ${supported} CACHE INTERNAL
        "GPU architectures supported by the compiler")
    mark_as_advanced(FORCE CAS_SUPPORTED_ARCHITECTURES)
    message(STATUS 
        "The CUDA compiler supports the following architectures: ${supported}")
    set(${output} ${CAS_SUPPORTED_ARCHITECTURES} PARENT_SCOPE)
endfunction()


# returns a list of GPU architectures present on the system
function(cas_get_onboard_architectures output)
    # Optional argument: disable warning. This is useful when calling this in a
    # function which already prints a warning.
    set(ENABLE_WARNING ON)
    if (ARGV1 EQUAL 0)
        set(ENABLE_WARNING OFF)
    endif()
    if(DEFINED CAS_ONBOARD_ARCHITECTURES)
        set(${output} "${CAS_ONBOARD_ARCHITECTURES}" PARENT_SCOPE)
        return()
    endif()
    set(detector_name "${PROJECT_BINARY_DIR}/CMakeFiles/cas_detector.cu")
    file(WRITE ${detector_name}
        "#include <iostream>\n"
        "int main() {"
        "  int n = 0;"
        "  cudaError_t r = cudaGetDeviceCount(&n);"
        "  if (r == cudaErrorNoDevice) return 0;"
        "  if (r != cudaSuccess) return 1;"
        "  char *sep = \"\";"
        "  for (int i = 0; i < n; ++i) {"
        "    cudaDeviceProp p;"
        "    if (cudaGetDeviceProperties(&p, i) == cudaSuccess) {"
        "      std::cout << sep << p.major << p.minor;"
        "      sep = \";\";"
        "    }"
        "  }"
        "  return 0;"
        "}")
    try_run(status unused
        "${PROJECT_BINARY_DIR}/CMakeFiles" "${detector_name}"
        RUN_OUTPUT_VARIABLE detected)
    if(status EQUAL 0)
        list(SORT detected)
        list(REMOVE_DUPLICATES detected)
        set(CAS_ONBOARD_ARCHITECTURES ${detected} CACHE INTERNAL
            "List of detected GPU architectures")
        mark_as_advanced(FORCE CAS_ONBOARD_ARCHITECTURES)
        message(STATUS
            "Detected GPU devices of the following architectures: ${detected}")
    elseif(ENABLE_WARNING)
        message(WARNING
            "GPU detection failed -- something seems to be wrong "
            "with the CUDA installation")
    endif()
    set(${output} "${CAS_ONBOARD_ARCHITECTURES}" PARENT_SCOPE)
endfunction()


# returns the list of GPU architectures associated to a specific GPU generation
function(cas_get_architectures_by_name name output)
    # add new name to compute capability bindings to this list as new GPU
    # generations get released
    set(  tesla_version 1)
    set(  fermi_version 2)
    set( kepler_version 3)
    set(maxwell_version 5)
    set( pascal_version 6)
    set(  volta_version "7(0|2)")
    set( turing_version 75)
    set( ampere_version 8)
    string(TOLOWER ${name} lower_name)
    if(NOT DEFINED ${lower_name}_version)
        message(FATAL_ERROR "${name} is not a valid GPU generation name")
    endif()
    cas_get_supported_architectures(architecture_list)
    list(FILTER architecture_list INCLUDE REGEX "^${${lower_name}_version}[0-9]?")
    set(${output} "${architecture_list}" PARENT_SCOPE)
endfunction()


# checks if the specified architecture specification is supported by the
# selected CUDA compiler
function(cas_is_supported_architecture_spec arch output)
    if (NOT (arch STREQUAL "") AND (arch MATCHES "${cas_spec_regex}"))
        set(code "${CMAKE_MATCH_1}")
        set(arch "${CMAKE_MATCH_3}")
        cas_get_supported_architectures(supported)
        list(FIND supported "${arch}" arch_supported)
        list(FIND supported "${code}" code_supported)
        if ((arch STREQUAL "" OR NOT (arch_supported EQUAL "-1")) AND 
            (code STREQUAL "" OR NOT (code_supported EQUAL "-1")))
            set(${output} TRUE PARENT_SCOPE)
        else()
            set(${output} FALSE PARENT_SCOPE)
        endif()
    else()
        message(FATAL_ERROR
            "${arch} is not a valid GPU architecture specification")
    endif()
endfunction()


# Adds or removes entries from a flag list using the ARCHITECTURES or
# UNSUPPORTED lists.
function(cas_update_flag_list flags_name mode)
    set(flags "${${flags_name}}")
    if(mode STREQUAL "ARCHITECTURES")
        foreach(spec IN LISTS ARGN)
            cas_is_supported_architecture_spec(${spec} is_arch_supported)
            if(NOT is_arch_supported)
                message(FATAL_ERROR
                    "${spec} is not an architecture specification supported by "
                    "the selected CUDA compiler")
            endif()
            string(REGEX MATCH "${cas_spec_regex}" unused "${spec}")
            set(code "${CMAKE_MATCH_1}")
            set(arch "${CMAKE_MATCH_3}")
            if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
                if(code STREQUAL "")
                    set(new_flag "sm_${arch}")
                else()
                    set(new_flag "sm_${code}")
                endif()
                list(APPEND flags "--cuda-gpu-arch=${new_flag}")
            else()
                if(arch STREQUAL "")
                    # virtual architecture not specified, use the one corresponding
                    # to the real architecture
                    set(arch "${code}")
                endif()
                if(code STREQUAL "")
                    # real architecture not specified, only generate PTX for the
                    # virtual architecture
                    set(new_flag "arch=compute_${arch},code=compute_${arch}")
                else()
                    # real architecture specified, generate CUBIN object for that
                    # architecture
                    set(new_flag "arch=compute_${arch},code=sm_${code}")
                endif()
                list(APPEND flags "--generate-code=${new_flag}")
            endif()
        endforeach()
    elseif(mode STREQUAL "UNSUPPORTED")
        foreach(arch IN LISTS ARGN)
            cas_get_supported_architectures(supported)
            if(NOT (supported MATCHES "${arch}"))
                continue()
            endif()
            foreach(flag IN ITEMS ${flags})
                if(flag MATCHES ".*${arch}.*")
                    message(WARNING
                        "Removing flag for unsupported architecture ${arch}: "
                        "${flag}")
                    list(REMOVE_ITEM flags "${flag}")
                endif()
            endforeach()
        endforeach()
    else()
        message(FATAL_ERROR 
            "Unknown mode ${mode} passed to cas_update_flag_list")
    endif()
    set(${flags_name} "${flags}" PARENT_SCOPE)
endfunction()


# Returns a list of flags that should be passed to the CUDA compiler to build
# for the passed architecture configurations.
function(cas_get_compiler_flags output)
    set(flags "")
    foreach(arg IN LISTS ARGN)
        if(arg STREQUAL "ARCHITECTURES" OR arg STREQUAL "UNSUPPORTED")
            set(stage ${arg})
            continue()
        endif()
        if(NOT DEFINED stage)
            message(FATAL_ERROR
                "cas_get_compiler_flags given unknown argument ${arg}")
        endif()
        if(arg STREQUAL "Auto")
            cas_get_onboard_architectures(detected 0)
            if(detected STREQUAL "")
                message(WARNING
                    "No GPUs detected on the system -- adding All flags "
                    "to the list of architectures")
                cas_get_supported_architectures(supported)
                cas_update_flag_list(flags ${stage} ${supported})
            endif()
            cas_update_flag_list(flags ${stage} ${detected})
        elseif(arg STREQUAL "All")
            cas_get_supported_architectures(supported)
            cas_update_flag_list(flags ${stage} ${supported})
        elseif(NOT (arg STREQUAL "") AND (arg MATCHES "${cas_spec_regex}"))
            cas_update_flag_list(flags ${stage} ${arg})
        elseif(NOT (arg STREQUAL ""))
            cas_get_architectures_by_name("${arg}" gen)
            cas_update_flag_list(flags ${stage} ${gen})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES flags)
    list(SORT flags)
    set(${output} "${flags}" PARENT_SCOPE)
endfunction()


# Sets the CUDA architectures that a target should be compiled for.
function(cas_target_cuda_architectures target)
    cas_get_compiler_flags(flags ${ARGN})
    target_compile_options(${target}
        PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${flags}>")
endfunction()

# It only outputs the compiler flags.
function(cas_variable_cuda_architectures variable)
    cas_get_compiler_flags(flags ${ARGN})
    set(${variable} "${flags}" PARENT_SCOPE)
endfunction()


function(cas_variable_cmake_cuda_architectures variable)
    cas_get_supported_architectures(supported_archs)
    if("${ARGN}" STREQUAL "All")
        set(archs "${supported_archs}")
    elseif("${ARGN}" STREQUAL "Auto")
        cas_get_onboard_architectures(onboard_archs)
        if (onboard_archs)
            set(archs "${onboard_archs}")
        else()
            set(archs "${supported_archs}")
        endif()
    else()
        set(archs)
        foreach(arch IN LISTS ARGN)
            if(arch MATCHES "${cas_spec_regex}")
                if(CMAKE_MATCH_1)
                    list(APPEND archs ${CMAKE_MATCH_1}-real)
                endif()
                if(CMAKE_MATCH_3)
                    list(APPEND archs ${CMAKE_MATCH_3}-virtual)
                endif()
            else()
                cas_get_architectures_by_name("${arch}" arch)
                list(APPEND archs ${arch})
            endif()
        endforeach()
    endif()
    set("${variable}" "${archs}" PARENT_SCOPE)
endfunction()

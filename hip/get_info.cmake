ginkgo_print_module_header(${detailed_log} "HIP")
ginkgo_print_module_footer(${detailed_log} "HIP variables:")
ginkgo_print_flags(${detailed_log} "CMAKE_HIP_FLAGS")
ginkgo_print_flags(${detailed_log} "CMAKE_HIP_COMPILER")
ginkgo_print_foreach_variable(${detailed_log}
    "HIP_VERSION;HIP_PATH;ROCM_PATH"
    "HIP_PLATFORM;HIP_ROOT_DIR;HIP_RUNTIME;HIPBLAS_PATH;HIPSPARSE_PATH"
    "HIPRAND_PATH;ROCRAND_PATH;HIP_CLANG_INCLUDE_PATH"
    "HIP_HIPCONFIG_EXECUTABLE")
ginkgo_print_module_footer(${detailed_log} "")

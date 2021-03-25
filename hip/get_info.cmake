ginkgo_print_module_header(${detailed_log} "HIP")
ginkgo_print_foreach_variable(${detailed_log}
    "GINKGO_HIPCONFIG_PATH;GINKGO_HIP_AMDGPU"
    "GINKGO_HIP_CLANG_COMPILER_FLAGS;GINKGO_HIP_NVCC_COMPILER_FLAGS"
    "GINKGO_HIP_THRUST_PATH;GINKGO_AMD_ARCH_FLAGS")
ginkgo_print_module_footer(${detailed_log} "HIP variables:")
ginkgo_print_foreach_variable(${detailed_log}
    "HIP_VERSION;HIP_COMPILER;HIP_PATH;ROCM_PATH"
    "HIP_PLATFORM;HIP_ROOT_DIR;HIP_RUNTIME;HIPBLAS_PATH;HIPSPARSE_PATH"
    "HIPRAND_PATH;ROCRAND_PATH;HIP_CLANG_INCLUDE_PATH;HIP_CLANG_PATH"
    "HIP_HIPCC_EXECUTABLE;HIP_HIPCONFIG_EXECUTABLE;HIP_HOST_COMPILATION_CPP")
ginkgo_print_flags(${detailed_log} "HIP_HIPCC_FLAGS")
ginkgo_print_flags(${detailed_log} "HIP_NVCC_FLAGS")
ginkgo_print_flags(${detailed_log} "HIP_CLANG_FLAGS")
ginkgo_print_module_footer(${detailed_log} "")

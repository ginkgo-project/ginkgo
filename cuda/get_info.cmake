# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_print_module_header(${detailed_log} "CUDA")
ginkgo_print_variable(${detailed_log} "CMAKE_CUDA_ARCHITECTURES")
ginkgo_print_variable(${detailed_log} "GINKGO_CUDA_COMPILER_FLAGS")
ginkgo_print_module_footer(${detailed_log} "CUDA variables:")
ginkgo_print_variable(${detailed_log} "CMAKE_CUDA_COMPILER")
ginkgo_print_variable(${detailed_log} "CMAKE_CUDA_COMPILER_VERSION")
ginkgo_print_flags(${detailed_log} "CMAKE_CUDA_FLAGS")
ginkgo_print_variable(${detailed_log} "CMAKE_CUDA_HOST_COMPILER")
ginkgo_print_variable(${detailed_log} "CUDAToolkit_LIBRARY_DIR")
ginkgo_print_module_footer(${detailed_log} "")

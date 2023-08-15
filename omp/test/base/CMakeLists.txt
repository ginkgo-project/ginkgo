# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_test(kernel_launch)
target_compile_definitions(omp_test_base_kernel_launch PRIVATE GKO_COMPILING_OMP)
target_link_libraries(omp_test_base_kernel_launch PRIVATE OpenMP::OpenMP_CXX)
ginkgo_create_test(index_set)

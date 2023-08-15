# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_dpcpp_test(executor)
ginkgo_create_dpcpp_test(dim3)
ginkgo_create_dpcpp_test(kernel_launch)
# set correct flags for kernel_launch.hpp
target_compile_definitions(dpcpp_test_base_kernel_launch PRIVATE GKO_COMPILING_DPCPP)

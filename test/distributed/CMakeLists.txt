# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_common_test(matrix_kernels DISABLE_EXECUTORS dpcpp)
ginkgo_create_common_test(partition_kernels DISABLE_EXECUTORS dpcpp)
ginkgo_create_common_test(vector_kernels DISABLE_EXECUTORS dpcpp)

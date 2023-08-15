# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_common_device_test(csr_kernels)
ginkgo_create_common_test(csr_kernels2)
ginkgo_create_common_test(coo_kernels)
ginkgo_create_common_test(dense_kernels)
ginkgo_create_common_test(diagonal_kernels)
ginkgo_create_common_test(ell_kernels)
ginkgo_create_common_test(fbcsr_kernels DISABLE_EXECUTORS dpcpp)
if (hipfft_FOUND)
    ginkgo_create_common_test(fft_kernels DISABLE_EXECUTORS dpcpp)
else()
    ginkgo_create_common_test(fft_kernels DISABLE_EXECUTORS dpcpp hip)
endif()
ginkgo_create_common_test(hybrid_kernels)
ginkgo_create_common_test(matrix)
ginkgo_create_common_test(sellp_kernels)
ginkgo_create_common_test(sparsity_csr_kernels)

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_hip_test(fbcsr_kernels)
if (hipfft_FOUND)
    ginkgo_create_hip_test(fft_kernels)
endif()

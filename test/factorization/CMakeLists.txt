# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_common_test(cholesky_kernels DISABLE_EXECUTORS dpcpp)
ginkgo_create_common_test(lu_kernels DISABLE_EXECUTORS dpcpp)
ginkgo_create_common_test(ic_kernels DISABLE_EXECUTORS dpcpp omp)
ginkgo_create_common_test(ilu_kernels DISABLE_EXECUTORS dpcpp omp)
ginkgo_create_common_test(par_ic_kernels)
ginkgo_create_common_test(par_ict_kernels)
ginkgo_create_common_test(par_ilu_kernels)
ginkgo_create_common_test(par_ilut_kernels DISABLE_EXECUTORS dpcpp)

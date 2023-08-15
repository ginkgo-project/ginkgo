# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_common_and_reference_test(schwarz MPI_SIZE 3 DISABLE_EXECUTORS dpcpp)

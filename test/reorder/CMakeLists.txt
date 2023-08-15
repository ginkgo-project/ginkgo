# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_common_test(amd DISABLE_EXECUTORS dpcpp)
if (GINKGO_HAVE_METIS)
    ginkgo_create_common_test(nested_dissection)
endif()

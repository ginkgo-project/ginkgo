# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_test(convergence)
if (GINKGO_HAVE_PAPI_SDE)
    ginkgo_create_test(papi ADDITIONAL_LIBRARIES PAPI::PAPI)
endif()

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_test(convergence)
ginkgo_create_test(logger)
if (GINKGO_HAVE_PAPI_SDE)
    ginkgo_create_test(papi ADDITIONAL_LIBRARIES PAPI::PAPI)
endif()
ginkgo_create_test(performance_hint)
ginkgo_create_test(profiler_hook)
ginkgo_create_test(record)
ginkgo_create_test(stream)

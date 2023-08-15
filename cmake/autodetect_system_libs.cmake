# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

if (NOT DEFINED GINKGO_BUILD_HWLOC)
    find_package(HWLOC 2.1)
endif()

if (NOT DEFINED GINKGO_BUILD_PAPI_SDE)
    find_package(PAPI 7.0.1.0 COMPONENTS sde)
endif()

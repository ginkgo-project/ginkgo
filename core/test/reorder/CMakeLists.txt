# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_test(amd)
if(GINKGO_HAVE_METIS)
    ginkgo_create_test(nested_dissection)
endif()
ginkgo_create_test(rcm)
ginkgo_create_test(scaled_reordered)

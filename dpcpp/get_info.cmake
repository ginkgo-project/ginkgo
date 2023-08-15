# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_print_module_header(${detailed_log} "DPCPP")
ginkgo_print_module_footer(${detailed_log} "DPCPP variables:")
ginkgo_print_variable(${detailed_log} "GINKGO_DPCPP_FLAGS")
ginkgo_print_variable(${detailed_log} "GINKGO_DPCPP_SINGLE_MODE")
ginkgo_print_module_footer(${detailed_log} "DPCPP environment variables:")
ginkgo_print_env_variable(${detailed_log} "SYCL_DEVICE_FILTER")
ginkgo_print_module_footer(${detailed_log} "")

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_hip_test(hip_executor)
ginkgo_create_test(index_set)
if(GINKGO_HAVE_HWLOC)
    find_package(NUMA REQUIRED)
    ginkgo_create_hip_test(hip_executor_topology ADDITIONAL_LIBRARIES NUMA::NUMA)
endif()
ginkgo_create_hip_test(kernel_launch)
# correct flags for kernel_launch.hpp are set in GINKGO_HIPCC_OPTIONS
ginkgo_create_hip_test(lin_op)
ginkgo_create_hip_test(math)
ginkgo_create_hip_test(memory)
# Only hcc needs the libraries. nvcc only requires the headers.
if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
    ginkgo_create_hip_test(exception_helpers ADDITIONAL_LIBRARIES roc::hipblas roc::hipsparse hip::hiprand roc::rocrand)
else()
    ginkgo_create_hip_test(exception_helpers)
endif()
ginkgo_create_hip_test(scoped_device_id)

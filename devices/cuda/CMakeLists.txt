# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_add_object_library(ginkgo_cuda_device
    executor.cpp)
if(GINKGO_BUILD_CUDA)
    target_compile_definitions(ginkgo_cuda_device PRIVATE GKO_COMPILING_CUDA_DEVICE)
endif()

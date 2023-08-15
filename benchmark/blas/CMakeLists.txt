# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_add_typed_benchmark_executables(blas "NO" blas.cpp)
if(GINKGO_BUILD_MPI)
    add_subdirectory(distributed)
endif()

// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <variant>

namespace gko {

struct reference_kernel {};
struct omp_kernel {};
struct cuda_kernel {};
struct hip_kernel {};
struct sycl_kernel {};

using cpu_kernel = std::variant<reference_kernel, omp_kernel>;
using cuda_hip_kernel = std::variant<cuda_kernel, hip_kernel>;
using any_kernel = std::variant<reference_kernel, omp_kernel, cuda_kernel,
                                hip_kernel, sycl_kernel>;


namespace kernels {
namespace cuda {


using device_kernel = cuda_kernel;


}
}  // namespace kernels


namespace kernels {
namespace hip {


using device_kernel = hip_kernel;


}
}  // namespace kernels
}  // namespace gko

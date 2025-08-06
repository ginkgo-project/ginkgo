// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_FACTORIZATION_PAR_ILUT_CONFIG_HIP_HPP_
#define GKO_COMMON_CUDA_HIP_FACTORIZATION_PAR_ILUT_CONFIG_HIP_HPP_

#include "common/cuda_hip/base/config.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace par_ilut_factorization {


constexpr int default_block_size = 512;


// subwarp sizes for add_candidates kernels
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


}  // namespace par_ilut_factorization
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

#endif  // GKO_COMMON_CUDA_HIP_FACTORIZATION_PAR_ILUT_CONFIG_HIP_HPP_

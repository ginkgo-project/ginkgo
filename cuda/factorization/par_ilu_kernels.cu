// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilu_kernels.hpp"


#include <ginkgo/core/matrix/coo.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/memory.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilu_factorization {


constexpr int default_block_size{512};


#include "common/cuda_hip/factorization/par_ilu_kernels.hpp.inc"


}  // namespace par_ilu_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

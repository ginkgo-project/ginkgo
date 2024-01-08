// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ic_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/memory.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ic factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ic_factorization {


constexpr int default_block_size = 512;


#include "common/cuda_hip/factorization/par_ic_kernels.hpp.inc"


}  // namespace par_ic_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

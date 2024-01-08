// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ic_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/memory.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The parallel ic factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ic_factorization {


constexpr int default_block_size = 512;


#include "common/cuda_hip/factorization/par_ic_kernels.hpp.inc"


}  // namespace par_ic_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko

// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/rcm_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/prefix_sum.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


#include "common/cuda_hip/reorder/rcm_kernels.hpp.inc"


}  // namespace rcm
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

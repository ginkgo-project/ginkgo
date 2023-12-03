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


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


#include "common/cuda_hip/reorder/rcm_kernels.hpp.inc"


}  // namespace rcm
}  // namespace hip
}  // namespace kernels
}  // namespace gko

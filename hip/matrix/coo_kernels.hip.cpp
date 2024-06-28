// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/coo_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/format_conversion.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/segment_scan.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;


#include "common/cuda_hip/matrix/coo_kernels.hpp.inc"


}  // namespace coo
}  // namespace hip
}  // namespace kernels
}  // namespace gko

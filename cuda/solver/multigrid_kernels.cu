// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/multigrid_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The MULTIGRID solver namespace.
 *
 * @ingroup multigrid
 */
namespace multigrid {


constexpr int default_block_size = 512;


#include "common/cuda_hip/solver/multigrid_kernels.hpp.inc"


}  // namespace multigrid
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

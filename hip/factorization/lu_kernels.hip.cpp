// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/lu_kernels.hpp"


#include <algorithm>
#include <memory>


#include <thrust/copy.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/allocator.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "hip/base/thrust.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/syncfree.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The LU namespace.
 *
 * @ingroup factor
 */
namespace lu_factorization {


constexpr static int default_block_size = 512;


#include "common/cuda_hip/factorization/lu_kernels.hpp.inc"


}  // namespace lu_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko

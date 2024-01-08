// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pgm_kernels.hpp"


#include <memory>


#include <cuda.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>


#include "cuda/base/thrust.cuh"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The PGM solver namespace.
 *
 * @ingroup pgm
 */
namespace pgm {


#include "common/cuda_hip/multigrid/pgm_kernels.hpp.inc"


}  // namespace pgm
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

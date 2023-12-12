// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_diagonal_kernels.hpp"


// Copyright (c) 2017-2023, the Ginkgo authors
#include <algorithm>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "cuda/base/batch_struct.hpp"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Diagonal matrix format namespace.
 * @ref Diagonal
 * @ingroup batch_diagonal
 */
namespace batch_diagonal {


constexpr auto default_block_size = 256;
constexpr int sm_oversubscription = 4;


template <typename T>
auto as_device_type(T ptr)
{
    return as_cuda_type(ptr);
}


// clang-format off

// NOTE: DO NOT CHANGE THE ORDERING OF THE INCLUDES

#include "common/cuda_hip/matrix/batch_diagonal_kernels.hpp.inc"


#include "common/cuda_hip/matrix/batch_diagonal_kernel_launcher.hpp.inc"

// clang-format on


}  // namespace batch_diagonal
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

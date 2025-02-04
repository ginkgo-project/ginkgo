// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#if GINKGO_BUILD_HIP
#include "../cuda_hip/batch_user_linop.hpp"
#endif


namespace gko {
namespace kernels {
namespace hip {
namespace batch_template {
namespace batch_user {


template <typename ValueType, typename UserOpView>
void apply(std::shared_ptr<const DefaultExecutor> exec, const UserOpView mat,
           batch::multi_vector::uniform_batch<const ValueType> b,
           batch::multi_vector::uniform_batch<ValueType> x)
{
#if GINKGO_BUILD_HIP
    auto num_rows = mat.num_rows;

    apply_kernel<<<mat.num_batch_items, get_num_threads_per_block(num_rows), 0,
                   exec->get_stream()>>>(mat, b, x);
#else
    GKO_NOT_IMPLEMENTED;
#endif
}


}  // namespace batch_user
}  // namespace batch_template
}  // namespace hip
}  // namespace kernels
}  // namespace gko

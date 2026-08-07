// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/cuda_hip/base/blas_bindings.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/pointer_mode_guard.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/multivector_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  matrix::view::dense<const ValueType> a,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> c)
{
    if (blas::is_supported<ValueType>::value) {
        auto handle = exec->get_blas_handle();
        if (c.size[0] > 0 && c.size[1] > 0) {
            if (a.size[1] > 0) {
                blas::pointer_mode_guard pm_guard(handle);
                auto alpha = one<ValueType>();
                auto beta = zero<ValueType>();
                blas::gemm(handle, BLAS_OP_N, BLAS_OP_N, c.size[1], c.size[0],
                           a.size[1], &alpha, b.values, b.stride, a.values,
                           a.stride, &beta, c.values, c.stride);
            } else {
                multivector::fill(exec, c, zero<ValueType>());
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           matrix::view::dense<const ValueType> alpha,
           matrix::view::dense<const ValueType> a,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<const ValueType> beta,
           matrix::view::dense<ValueType> c)
{
    if (blas::is_supported<ValueType>::value) {
        if (c.size[0] > 0 && c.size[1] > 0) {
            if (a.size[1] > 0) {
                blas::gemm(exec->get_blas_handle(), BLAS_OP_N, BLAS_OP_N,
                           c.size[1], c.size[0], a.size[1], alpha.values,
                           b.values, b.stride, a.values, a.stride, beta.values,
                           c.values, c.stride);
            } else {
                multivector::scale(exec, beta, c);
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


}  // namespace dense
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

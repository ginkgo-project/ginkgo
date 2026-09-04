// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/multivector_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/cuda_hip/base/blas_bindings.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/pointer_mode_guard.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "core/base/utils.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The MultiVector matrix format namespace.
 *
 * @ingroup dense
 */
namespace multivector {


template <typename ValueType>
void compute_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                          matrix::view::dense<const ValueType> x,
                          matrix::view::dense<const ValueType> y,
                          matrix::view::dense<ValueType> result,
                          array<char>& tmp)
{
    if (x.size[1] == 1 && y.size[1] == 1) {
        if (blas::is_supported<ValueType>::value) {
            auto handle = exec->get_blas_handle();
            blas::dot(handle, x.size[0], x.values, x.stride, y.values, y.stride,
                      result.values);
        } else {
            compute_dot(exec, x, y, result, tmp);
        }
    } else {
        compute_dot(exec, x, y, result, tmp);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_conj_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                               matrix::view::dense<const ValueType> x,
                               matrix::view::dense<const ValueType> y,
                               matrix::view::dense<ValueType> result,
                               array<char>& tmp)
{
    if (x.size[1] == 1 && y.size[1] == 1) {
        if (blas::is_supported<ValueType>::value) {
            auto handle = exec->get_blas_handle();
            blas::conj_dot(handle, x.size[0], x.values, x.stride, y.values,
                           y.stride, result.values);
        } else {
            compute_conj_dot(exec, x, y, result, tmp);
        }
    } else {
        compute_conj_dot(exec, x, y, result, tmp);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm2_dispatch(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> x,
    matrix::view::dense<remove_complex<ValueType>> result, array<char>& tmp)
{
    if (x.size[1] == 1) {
        if (blas::is_supported<ValueType>::value) {
            auto handle = exec->get_blas_handle();
            blas::norm2(handle, x.size[0], x.values, x.stride, result.values);
        } else {
            compute_norm2(exec, x, result, tmp);
        }
    } else {
        compute_norm2(exec, x, result, tmp);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_NORM2_DISPATCH_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               matrix::view::dense<const ValueType> orig,
               matrix::view::dense<ValueType> trans)
{
    if (blas::is_supported<ValueType>::value) {
        auto handle = exec->get_blas_handle();
        if (orig.size[0] > 0 && orig.size[1] > 0) {
            blas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            blas::geam(handle, BLAS_OP_T, BLAS_OP_N, orig.size[0], orig.size[1],
                       &alpha, orig.values, orig.stride, &beta, trans.values,
                       trans.stride, trans.values, trans.stride);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,
                    matrix::view::dense<const ValueType> orig,
                    matrix::view::dense<ValueType> trans)
{
    if (blas::is_supported<ValueType>::value) {
        auto handle = exec->get_blas_handle();
        if (orig.size[0] > 0 && orig.size[1] > 0) {
            blas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            blas::geam(handle, BLAS_OP_C, BLAS_OP_N, orig.size[0], orig.size[1],
                       &alpha, orig.values, orig.stride, &beta, trans.values,
                       trans.stride, trans.values, trans.stride);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_CONJ_TRANSPOSE_KERNEL);


}  // namespace multivector
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

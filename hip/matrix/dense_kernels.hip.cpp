// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "common/cuda_hip/base/blas_bindings.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/pointer_mode_guard.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


constexpr int default_block_size = 512;


#include "common/cuda_hip/matrix/dense_kernels.hpp.inc"


template <typename ValueType>
void compute_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* x,
                          const matrix::Dense<ValueType>* y,
                          matrix::Dense<ValueType>* result, array<char>& tmp)
{
    if (x->get_size()[1] == 1 && y->get_size()[1] == 1) {
        if (blas::is_supported<ValueType>::value) {
            auto handle = exec->get_blas_handle();
            blas::dot(handle, x->get_size()[0], x->get_const_values(),
                      x->get_stride(), y->get_const_values(), y->get_stride(),
                      result->get_values());
        } else {
            compute_dot(exec, x, y, result, tmp);
        }
    } else {
        compute_dot(exec, x, y, result, tmp);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_conj_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                               const matrix::Dense<ValueType>* x,
                               const matrix::Dense<ValueType>* y,
                               matrix::Dense<ValueType>* result,
                               array<char>& tmp)
{
    if (x->get_size()[1] == 1 && y->get_size()[1] == 1) {
        if (blas::is_supported<ValueType>::value) {
            auto handle = exec->get_blas_handle();
            blas::conj_dot(handle, x->get_size()[0], x->get_const_values(),
                           x->get_stride(), y->get_const_values(),
                           y->get_stride(), result->get_values());
        } else {
            compute_conj_dot(exec, x, y, result, tmp);
        }
    } else {
        compute_conj_dot(exec, x, y, result, tmp);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm2_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Dense<ValueType>* x,
                            matrix::Dense<remove_complex<ValueType>>* result,
                            array<char>& tmp)
{
    if (x->get_size()[1] == 1) {
        if (blas::is_supported<ValueType>::value) {
            auto handle = exec->get_blas_handle();
            blas::norm2(handle, x->get_size()[0], x->get_const_values(),
                        x->get_stride(), result->get_values());
        } else {
            compute_norm2(exec, x, result, tmp);
        }
    } else {
        compute_norm2(exec, x, result, tmp);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL);


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* a,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* c)
{
    if (blas::is_supported<ValueType>::value) {
        auto handle = exec->get_blas_handle();
        if (c->get_size()[0] > 0 && c->get_size()[1] > 0) {
            if (a->get_size()[1] > 0) {
                blas::pointer_mode_guard pm_guard(handle);
                auto alpha = one<ValueType>();
                auto beta = zero<ValueType>();
                blas::gemm(handle, BLAS_OP_N, BLAS_OP_N, c->get_size()[1],
                           c->get_size()[0], a->get_size()[1], &alpha,
                           b->get_const_values(), b->get_stride(),
                           a->get_const_values(), a->get_stride(), &beta,
                           c->get_values(), c->get_stride());
            } else {
                dense::fill(exec, c, zero<ValueType>());
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* a, const matrix::Dense<ValueType>* b,
           const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* c)
{
    if (blas::is_supported<ValueType>::value) {
        if (c->get_size()[0] > 0 && c->get_size()[1] > 0) {
            if (a->get_size()[1] > 0) {
                blas::gemm(exec->get_blas_handle(), BLAS_OP_N, BLAS_OP_N,
                           c->get_size()[1], c->get_size()[0], a->get_size()[1],
                           alpha->get_const_values(), b->get_const_values(),
                           b->get_stride(), a->get_const_values(),
                           a->get_stride(), beta->get_const_values(),
                           c->get_values(), c->get_stride());
            } else {
                dense::scale(exec, beta, c);
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* orig,
               matrix::Dense<ValueType>* trans)
{
    if (blas::is_supported<ValueType>::value) {
        auto handle = exec->get_blas_handle();
        if (orig->get_size()[0] > 0 && orig->get_size()[1] > 0) {
            blas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            blas::geam(handle, BLAS_OP_T, BLAS_OP_N, orig->get_size()[0],
                       orig->get_size()[1], &alpha, orig->get_const_values(),
                       orig->get_stride(), &beta, trans->get_const_values(),
                       trans->get_stride(), trans->get_values(),
                       trans->get_stride());
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* orig,
                    matrix::Dense<ValueType>* trans)
{
    if (blas::is_supported<ValueType>::value) {
        auto handle = exec->get_blas_handle();
        if (orig->get_size()[0] > 0 && orig->get_size()[1] > 0) {
            blas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            blas::geam(handle, BLAS_OP_C, BLAS_OP_N, orig->get_size()[0],
                       orig->get_size()[1], &alpha, orig->get_const_values(),
                       orig->get_stride(), &beta, trans->get_const_values(),
                       trans->get_stride(), trans->get_values(),
                       trans->get_stride());
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace hip
}  // namespace kernels
}  // namespace gko

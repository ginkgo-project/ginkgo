/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/dense_kernels.hpp"


#include <hip/hip_runtime.h>


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


#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
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
        if (hipblas::is_supported<ValueType>::value) {
            auto handle = exec->get_hipblas_handle();
            hipblas::dot(handle, x->get_size()[0], x->get_const_values(),
                         x->get_stride(), y->get_const_values(),
                         y->get_stride(), result->get_values());
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
        if (hipblas::is_supported<ValueType>::value) {
            auto handle = exec->get_hipblas_handle();
            hipblas::conj_dot(handle, x->get_size()[0], x->get_const_values(),
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
        if (hipblas::is_supported<ValueType>::value) {
            auto handle = exec->get_hipblas_handle();
            hipblas::norm2(handle, x->get_size()[0], x->get_const_values(),
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
    if (hipblas::is_supported<ValueType>::value) {
        auto handle = exec->get_hipblas_handle();
        if (c->get_size()[0] > 0 && c->get_size()[1] > 0) {
            if (a->get_size()[1] > 0) {
                hipblas::pointer_mode_guard pm_guard(handle);
                auto alpha = one<ValueType>();
                auto beta = zero<ValueType>();
                hipblas::gemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              c->get_size()[1], c->get_size()[0],
                              a->get_size()[1], &alpha, b->get_const_values(),
                              b->get_stride(), a->get_const_values(),
                              a->get_stride(), &beta, c->get_values(),
                              c->get_stride());
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
    if (hipblas::is_supported<ValueType>::value) {
        if (c->get_size()[0] > 0 && c->get_size()[1] > 0) {
            if (a->get_size()[1] > 0) {
                hipblas::gemm(
                    exec->get_hipblas_handle(), HIPBLAS_OP_N, HIPBLAS_OP_N,
                    c->get_size()[1], c->get_size()[0], a->get_size()[1],
                    alpha->get_const_values(), b->get_const_values(),
                    b->get_stride(), a->get_const_values(), a->get_stride(),
                    beta->get_const_values(), c->get_values(), c->get_stride());
            } else {
                dense::scale(exec, beta, c);
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    const int64* row_ptrs,
                    matrix::Coo<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto stride = source->get_stride();

    const auto grid_dim =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (grid_dim > 0) {
        hipLaunchKernelGGL(kernel::fill_in_coo, grid_dim, default_block_size, 0,
                           0, num_rows, num_cols, stride,
                           as_hip_type(source->get_const_values()), row_ptrs,
                           row_idxs, col_idxs, as_hip_type(values));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto stride = source->get_stride();

    const auto grid_dim =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (grid_dim > 0) {
        hipLaunchKernelGGL(
            kernel::fill_in_csr, grid_dim, default_block_size, 0, 0, num_rows,
            num_cols, stride, as_hip_type(source->get_const_values()),
            as_hip_type(row_ptrs), as_hip_type(col_idxs), as_hip_type(values));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Ell<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto max_nnz_per_row = result->get_num_stored_elements_per_row();

    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto source_stride = source->get_stride();
    auto result_stride = result->get_stride();

    const auto grid_dim =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (grid_dim > 0) {
        hipLaunchKernelGGL(
            kernel::fill_in_ell, grid_dim, default_block_size, 0, 0, num_rows,
            num_cols, source_stride, as_hip_type(source->get_const_values()),
            max_nnz_per_row, result_stride, col_idxs, as_hip_type(values));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Dense<ValueType>* source,
                      matrix::Fbcsr<ValueType, IndexType>* result)
{
    const auto num_block_rows = result->get_num_block_rows();
    if (num_block_rows > 0) {
        const auto num_blocks =
            ceildiv(num_block_rows, default_block_size / config::warp_size);
        kernel::convert_to_fbcsr<<<num_blocks, default_block_size>>>(
            num_block_rows, result->get_num_block_cols(), source->get_stride(),
            result->get_block_size(), as_hip_type(source->get_const_values()),
            result->get_const_row_ptrs(), result->get_col_idxs(),
            as_hip_type(result->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzero_blocks_per_row(std::shared_ptr<const DefaultExecutor> exec,
                                  const matrix::Dense<ValueType>* source,
                                  int bs, IndexType* result)
{
    const auto num_block_rows = source->get_size()[0] / bs;
    const auto num_block_cols = source->get_size()[1] / bs;
    if (num_block_rows > 0) {
        const auto num_blocks =
            ceildiv(num_block_rows, default_block_size / config::warp_size);
        kernel::
            count_nonzero_blocks_per_row<<<num_blocks, default_block_size>>>(
                num_block_rows, num_block_cols, source->get_stride(), bs,
                as_hip_type(source->get_const_values()), result);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType>* source,
                       const int64* coo_row_ptrs,
                       matrix::Hybrid<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto ell_max_nnz_per_row =
        result->get_ell_num_stored_elements_per_row();
    const auto source_stride = source->get_stride();
    const auto ell_stride = result->get_ell_stride();
    auto ell_col_idxs = result->get_ell_col_idxs();
    auto ell_values = result->get_ell_values();
    auto coo_row_idxs = result->get_coo_row_idxs();
    auto coo_col_idxs = result->get_coo_col_idxs();
    auto coo_values = result->get_coo_values();

    auto grid_dim = ceildiv(num_rows, default_block_size / config::warp_size);
    if (grid_dim > 0) {
        hipLaunchKernelGGL(kernel::fill_in_hybrid, grid_dim, default_block_size,
                           0, 0, num_rows, num_cols, source_stride,
                           as_hip_type(source->get_const_values()),
                           ell_max_nnz_per_row, ell_stride, ell_col_idxs,
                           as_hip_type(ell_values), coo_row_ptrs, coo_row_idxs,
                           coo_col_idxs, as_hip_type(coo_values));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Dense<ValueType>* source,
                      matrix::Sellp<ValueType, IndexType>* result)
{
    const auto stride = source->get_stride();
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];

    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_sets = result->get_slice_sets();

    const auto slice_size = result->get_slice_size();
    const auto stride_factor = result->get_stride_factor();

    auto grid_dim = ceildiv(num_rows, default_block_size / config::warp_size);
    if (grid_dim > 0) {
        hipLaunchKernelGGL(kernel::fill_in_sellp, grid_dim, default_block_size,
                           0, 0, num_rows, num_cols, slice_size, stride,
                           as_hip_type(source->get_const_values()),
                           as_hip_type(slice_sets), as_hip_type(col_idxs),
                           as_hip_type(vals));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Dense<ValueType>* source,
                             matrix::SparsityCsr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();

    auto stride = source->get_stride();

    const auto grid_dim =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (grid_dim > 0) {
        hipLaunchKernelGGL(kernel::fill_in_sparsity_csr, grid_dim,
                           default_block_size, 0, 0, num_rows, num_cols, stride,
                           as_hip_type(source->get_const_values()),
                           as_hip_type(row_ptrs), as_hip_type(col_idxs));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* orig,
               matrix::Dense<ValueType>* trans)
{
    if (hipblas::is_supported<ValueType>::value) {
        auto handle = exec->get_hipblas_handle();
        if (orig->get_size()[0] > 0 && orig->get_size()[1] > 0) {
            hipblas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            hipblas::geam(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                          orig->get_size()[0], orig->get_size()[1], &alpha,
                          orig->get_const_values(), orig->get_stride(), &beta,
                          trans->get_const_values(), trans->get_stride(),
                          trans->get_values(), trans->get_stride());
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
    if (hipblas::is_supported<ValueType>::value) {
        auto handle = exec->get_hipblas_handle();
        if (orig->get_size()[0] > 0 && orig->get_size()[1] > 0) {
            hipblas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            hipblas::geam(handle, HIPBLAS_OP_C, HIPBLAS_OP_N,
                          orig->get_size()[0], orig->get_size()[1], &alpha,
                          orig->get_const_values(), orig->get_stride(), &beta,
                          trans->get_values(), trans->get_stride(),
                          trans->get_values(), trans->get_stride());
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

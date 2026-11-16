// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ic_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/memory.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/factorization/factorization_helpers.hpp"
#include "common/unified/base/arithmetic_accessor.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The parallel ic factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ic_factorization {


constexpr int default_block_size = 512;


namespace kernel {


template <typename ValueType, typename IndexType, typename AAccessor>
__global__ __launch_bounds__(default_block_size) void ic_sweep(
    const IndexType* __restrict__ a_row_idxs,
    const IndexType* __restrict__ a_col_idxs, AAccessor a_vals,
    const IndexType* __restrict__ l_row_ptrs,
    const IndexType* __restrict__ l_col_idxs, ValueType* __restrict__ l_vals,
    IndexType l_nnz)
{
    using arithmetic_type = arithmetic_type_t<AAccessor>;
    const auto l_nz = thread::get_thread_id_flat<IndexType>();
    if (l_nz >= l_nnz) {
        return;
    }
    const auto row = a_row_idxs[l_nz];
    const auto col = l_col_idxs[l_nz];
    const auto a_val = a_vals[l_nz];
    auto l_row_begin = l_row_ptrs[row];
    const auto l_row_end = l_row_ptrs[row + 1];
    auto lh_col_begin = l_row_ptrs[col];
    const auto lh_col_end = l_row_ptrs[col + 1];
    arithmetic_type sum{};
    auto last_entry = col;
    while (l_row_begin < l_row_end && lh_col_begin < lh_col_end) {
        auto l_col = l_col_idxs[l_row_begin];
        auto lh_row = l_col_idxs[lh_col_begin];
        if (l_col == lh_row && l_col < last_entry) {
            sum += static_cast<arithmetic_type>(
                       load_relaxed(l_vals + l_row_begin)) *
                   conj(static_cast<arithmetic_type>(
                       load_relaxed(l_vals + lh_col_begin)));
        }
        l_row_begin += l_col <= lh_row;
        lh_col_begin += l_col >= lh_row;
    }
    auto to_write =
        row == col ? sqrt(a_val - sum)
                   : (a_val - sum) / static_cast<arithmetic_type>(load_relaxed(
                                         l_vals + (l_row_ptrs[col + 1] - 1)));
    if (is_finite(to_write)) {
        store_relaxed(l_vals + l_nz, static_cast<ValueType>(to_write));
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void init_factor(std::shared_ptr<const DefaultExecutor> exec,
                 matrix::view::csr<ValueType, IndexType> l)
{
    namespace helpers = factorization::helpers;
    auto num_rows = l.size[0];
    auto num_blocks = ceildiv(num_rows, helpers::default_block_size);
    auto l_row_ptrs = l.row_ptrs;
    const auto l_vals = arithmetic<ValueType>(l);
    if (num_blocks > 0) {
        helpers::ic_init<<<num_blocks, helpers::default_block_size, 0,
                           exec->get_stream()>>>(
            l_row_ptrs, acc::as_device_range(l_vals), num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    size_type iterations,
                    matrix::view::coo<const ValueType, const IndexType> a_lower,
                    matrix::view::csr<ValueType, IndexType> l)
{
    iterations = (iterations == 0) ? 10 : iterations;
    auto nnz = l.num_stored_elements;
    auto num_blocks = ceildiv(nnz, default_block_size);
    const auto a_vals =
        arithmetic<ValueType>(a_lower.values, a_lower.num_stored_elements);
    if (num_blocks > 0) {
#ifdef GKO_COMPILING_HIP
        if constexpr (sizeof(remove_complex<ValueType>) == sizeof(int16)) {
            // HIP does not support 16bit atomic operation
            GKO_NOT_SUPPORTED(a_lower);
        } else
#endif
        {
            for (size_type i = 0; i < iterations; ++i) {
                kernel::ic_sweep<<<num_blocks, default_block_size, 0,
                                   exec->get_stream()>>>(
                    a_lower.row_idxs, a_lower.col_idxs,
                    acc::as_device_range(a_vals), l.row_ptrs, l.col_idxs,
                    as_device_type(l.values),
                    static_cast<IndexType>(l.num_stored_elements));
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ic_factorization
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

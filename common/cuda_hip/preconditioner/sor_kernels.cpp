// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/sor_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/factorization/factorization_helpers.hpp"
#include "common/unified/base/arithmetic_accessor.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace sor {
namespace kernel {


/**
 * Multiplies the diagonal entry of each row of L, which is stored last in the
 * row, by diag_scale.
 */
template <typename IndexType, typename ValueAccessor, typename ScalarType>
__global__
__launch_bounds__(factorization::helpers::default_block_size) void scale_weighted_l(
    size_type num_rows, const IndexType* __restrict__ l_row_ptrs,
    ValueAccessor l_vals, ScalarType diag_scale)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row < num_rows) {
        const auto diag_idx = l_row_ptrs[row + 1] - 1;
        l_vals[diag_idx] *= diag_scale;
    }
}


/**
 * Multiplies the diagonal entry of each row of U, which is stored first in the
 * row, by diag_scale, and the off-diagonal entries by off_diag_scale1 and then
 * by off_diag_scale2.
 */
template <typename IndexType, typename ValueAccessor, typename ScalarType>
__global__
__launch_bounds__(factorization::helpers::default_block_size) void scale_weighted_u(
    size_type num_rows, const IndexType* __restrict__ u_row_ptrs,
    ValueAccessor u_vals, ScalarType diag_scale, ScalarType off_diag_scale1,
    ScalarType off_diag_scale2)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row < num_rows) {
        const auto begin = u_row_ptrs[row];
        const auto end = u_row_ptrs[row + 1];
        u_vals[begin] *= diag_scale;
        for (auto nz = begin + 1; nz < end; ++nz) {
            u_vals[nz] = u_vals[nz] * off_diag_scale1 * off_diag_scale2;
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void initialize_weighted_l(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> system_matrix,
    remove_complex<ValueType> weight,
    matrix::view::csr<ValueType, IndexType> l_mtx)
{
    const size_type num_rows{system_matrix.size[0]};
    const auto block_size = factorization::helpers::default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_rows, static_cast<size_type>(block_size)));

    const auto l_vals = arithmetic<ValueType>(l_mtx);
    if (grid_dim > 0) {
        using namespace gko::factorization;

        // the weighting is computed separately, so that the copy stays exact
        factorization::helpers::
            initialize_l<<<grid_dim, block_size, 0, exec->get_stream()>>>(
                num_rows, system_matrix.row_ptrs, system_matrix.col_idxs,
                as_device_type(system_matrix.values), l_mtx.row_ptrs,
                l_mtx.col_idxs, as_device_type(l_mtx.values),
                triangular_mtx_closure(identity{}, identity{}));
        kernel::
            scale_weighted_l<<<grid_dim, block_size, 0, exec->get_stream()>>>(
                num_rows, l_mtx.row_ptrs, acc::as_device_range(l_vals),
                arithmetic<ValueType>(one(weight) / weight));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L);


template <typename ValueType, typename IndexType>
void initialize_weighted_l_u(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> system_matrix,
    remove_complex<ValueType> weight,
    matrix::view::csr<ValueType, IndexType> l_mtx,
    matrix::view::csr<ValueType, IndexType> u_mtx)
{
    const size_type num_rows{system_matrix.size[0]};
    const auto block_size = factorization::helpers::default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_rows, static_cast<size_type>(block_size)));

    const auto inv_weight = one(weight) / weight;
    const auto inv_two_minus_weight =
        one(weight) / (static_cast<remove_complex<ValueType>>(2.0) - weight);

    const auto l_vals = arithmetic<ValueType>(l_mtx);
    const auto u_vals = arithmetic<ValueType>(u_mtx);
    if (grid_dim > 0) {
        using namespace gko::factorization;

        // the weighting is computed separately, so that the copy stays exact
        factorization::helpers::
            initialize_l_u<<<grid_dim, block_size, 0, exec->get_stream()>>>(
                num_rows, system_matrix.row_ptrs, system_matrix.col_idxs,
                as_device_type(system_matrix.values), l_mtx.row_ptrs,
                l_mtx.col_idxs, as_device_type(l_mtx.values), u_mtx.row_ptrs,
                u_mtx.col_idxs, as_device_type(u_mtx.values),
                triangular_mtx_closure(identity{}, identity{}),
                triangular_mtx_closure(identity{}, identity{}));
        kernel::
            scale_weighted_l<<<grid_dim, block_size, 0, exec->get_stream()>>>(
                num_rows, l_mtx.row_ptrs, acc::as_device_range(l_vals),
                arithmetic<ValueType>(inv_weight));
        kernel::
            scale_weighted_u<<<grid_dim, block_size, 0, exec->get_stream()>>>(
                num_rows, u_mtx.row_ptrs, acc::as_device_range(u_vals),
                arithmetic<ValueType>(inv_two_minus_weight),
                arithmetic<ValueType>(weight),
                arithmetic<ValueType>(inv_two_minus_weight));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L_U);


}  // namespace sor
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

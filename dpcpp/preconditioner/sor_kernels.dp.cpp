// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/sor_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "dpcpp/factorization/factorization_helpers.dp.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {
namespace sor {


constexpr int default_block_size{256};


template <typename ValueType, typename IndexType>
void initialize_weighted_l(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    remove_complex<ValueType> weight, matrix::Csr<ValueType, IndexType>* l_mtx)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const dim3 block_size{default_block_size, 1, 1};
    const dim3 grid_dim{static_cast<uint32>(ceildiv(
                            num_rows, static_cast<size_type>(block_size.x))),
                        1, 1};

    auto inv_weight = one(weight) / weight;
    const auto in_row_ptrs = system_matrix->get_const_row_ptrs();
    const auto in_col_idxs = system_matrix->get_const_col_idxs();
    const auto in_values = system_matrix->get_const_values();
    const auto l_row_ptrs = l_mtx->get_const_row_ptrs();
    const auto l_col_idxs = l_mtx->get_col_idxs();
    const auto l_values = l_mtx->get_values();

    exec->get_queue()->parallel_for(
        sycl_nd_range(grid_dim, block_size), [=](sycl::nd_item<3> item_ct1) {
            factorization::helpers::initialize_l(
                num_rows, in_row_ptrs, in_col_idxs, in_values, l_row_ptrs,
                l_col_idxs, l_values,
                factorization::helpers::triangular_mtx_closure(
                    [inv_weight](auto val) { return val * inv_weight; },
                    factorization::helpers::identity{}),
                item_ct1);
        });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L);


template <typename ValueType, typename IndexType>
void initialize_weighted_l_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    remove_complex<ValueType> weight, matrix::Csr<ValueType, IndexType>* l_mtx,
    matrix::Csr<ValueType, IndexType>* u_mtx)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const dim3 block_size{default_block_size, 1, 1};
    const dim3 grid_dim{static_cast<uint32>(ceildiv(
                            num_rows, static_cast<size_type>(block_size.x))),
                        1, 1};

    auto inv_weight = one(weight) / weight;
    auto inv_two_minus_weight =
        one(weight) / (static_cast<remove_complex<ValueType>>(2.0) - weight);

    const auto in_row_ptrs = system_matrix->get_const_row_ptrs();
    const auto in_col_idxs = system_matrix->get_const_col_idxs();
    const auto in_values = system_matrix->get_const_values();
    const auto l_row_ptrs = l_mtx->get_const_row_ptrs();
    const auto l_col_idxs = l_mtx->get_col_idxs();
    const auto l_values = l_mtx->get_values();
    const auto u_row_ptrs = u_mtx->get_const_row_ptrs();
    const auto u_col_idxs = u_mtx->get_col_idxs();
    const auto u_values = u_mtx->get_values();

    exec->get_queue()->parallel_for(
        sycl_nd_range(grid_dim, block_size), [=](sycl::nd_item<3> item_ct1) {
            factorization::helpers::initialize_l_u(
                num_rows, in_row_ptrs, in_col_idxs, in_values, l_row_ptrs,
                l_col_idxs, l_values, u_row_ptrs, u_col_idxs, u_values,
                factorization::helpers::triangular_mtx_closure(
                    [inv_weight](auto val) { return val * inv_weight; },
                    factorization::helpers::identity{}),
                factorization::helpers::triangular_mtx_closure(
                    [inv_two_minus_weight](auto val) {
                        return val * inv_two_minus_weight;
                    },
                    [weight, inv_two_minus_weight](auto val) {
                        return val * weight * inv_two_minus_weight;
                    }),
                item_ct1);
        });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L_U);


}  // namespace sor
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko

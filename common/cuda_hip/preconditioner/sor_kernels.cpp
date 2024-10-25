// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/sor_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "common/cuda_hip/factorization/factorization_helpers.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace sor {


template <typename ValueType, typename IndexType>
void initialize_weighted_l(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    remove_complex<ValueType> weight, matrix::Csr<ValueType, IndexType>* l_mtx)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const auto block_size = factorization::helpers::default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_rows, static_cast<size_type>(block_size)));

    auto inv_weight = as_device_type(one(weight) / weight);

    if (grid_dim > 0) {
        using namespace gko::factorization;

        factorization::helpers::
            initialize_l<<<grid_dim, block_size, 0, exec->get_stream()>>>(
                num_rows, system_matrix->get_const_row_ptrs(),
                system_matrix->get_const_col_idxs(),
                as_device_type(system_matrix->get_const_values()),
                l_mtx->get_const_row_ptrs(), l_mtx->get_col_idxs(),
                as_device_type(l_mtx->get_values()),
                triangular_mtx_closure(
                    [inv_weight] __device__(auto val) {
                        return val * inv_weight;
                    },
                    identity{}));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L);


template <typename ValueType, typename IndexType>
void initialize_weighted_l_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    remove_complex<ValueType> weight, matrix::Csr<ValueType, IndexType>* l_mtx,
    matrix::Csr<ValueType, IndexType>* u_mtx)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const auto block_size = factorization::helpers::default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_rows, static_cast<size_type>(block_size)));

    auto inv_weight = as_device_type(one(weight) / weight);
    auto inv_two_minus_weight = as_device_type(
        one(weight) / (static_cast<remove_complex<ValueType>>(2.0) - weight));
    auto d_weight = as_device_type(weight);

    if (grid_dim > 0) {
        using namespace gko::factorization;

        factorization::helpers::
            initialize_l_u<<<grid_dim, block_size, 0, exec->get_stream()>>>(
                num_rows, system_matrix->get_const_row_ptrs(),
                system_matrix->get_const_col_idxs(),
                as_device_type(system_matrix->get_const_values()),
                l_mtx->get_const_row_ptrs(), l_mtx->get_col_idxs(),
                as_device_type(l_mtx->get_values()),
                u_mtx->get_const_row_ptrs(), u_mtx->get_col_idxs(),
                as_device_type(u_mtx->get_values()),
                triangular_mtx_closure(
                    [inv_weight] __device__(auto val) {
                        return val * inv_weight;
                    },
                    identity{}),
                triangular_mtx_closure(
                    [inv_two_minus_weight] __device__(auto val) {
                        return val * inv_two_minus_weight;
                    },
                    [d_weight, inv_two_minus_weight] __device__(auto val) {
                        return val * d_weight * inv_two_minus_weight;
                    }));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L_U);


}  // namespace sor
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

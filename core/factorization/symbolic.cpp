// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/symbolic.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/base/allocator.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace factorization {
namespace {


GKO_REGISTER_OPERATION(symbolic_count, cholesky::symbolic_count);
GKO_REGISTER_OPERATION(symbolic, cholesky::symbolic_factorize);
GKO_REGISTER_OPERATION(build_lookup_offsets, csr::build_lookup_offsets);
GKO_REGISTER_OPERATION(build_lookup, csr::build_lookup);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(symbolic_factorize_simple,
                       lu_factorization::symbolic_factorize_simple);
GKO_REGISTER_OPERATION(symbolic_factorize_simple_finalize,
                       lu_factorization::symbolic_factorize_simple_finalize);
GKO_REGISTER_HOST_OPERATION(compute_elim_forest, compute_elim_forest);


}  // namespace


template <typename ValueType, typename IndexType>
void symbolic_cholesky(
    const matrix::Csr<ValueType, IndexType>* mtx, bool symmetrize,
    std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors,
    std::unique_ptr<elimination_forest<IndexType>>& forest)
{
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    GKO_ASSERT_IS_SQUARE_MATRIX(mtx);
    const auto exec = mtx->get_executor();
    const auto host_exec = exec->get_master();
    exec->run(make_compute_elim_forest(mtx, forest));
    const auto num_rows = mtx->get_size()[0];
    array<IndexType> row_ptrs{exec, num_rows + 1};
    array<IndexType> tmp{exec};
    exec->run(make_symbolic_count(mtx, *forest, row_ptrs.get_data(), tmp));
    exec->run(make_prefix_sum_nonnegative(row_ptrs.get_data(), num_rows + 1));
    const auto factor_nnz =
        static_cast<size_type>(get_element(row_ptrs, num_rows));
    factors = matrix_type::create(
        exec, mtx->get_size(), array<ValueType>{exec, factor_nnz},
        array<IndexType>{exec, factor_nnz}, std::move(row_ptrs));
    exec->run(make_symbolic(mtx, *forest, factors.get(), tmp));
    factors->sort_by_column_index();
    if (symmetrize) {
        auto lt_factor = as<matrix_type>(factors->transpose());
        const auto scalar =
            initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
        const auto id = matrix::Identity<ValueType>::create(exec, num_rows);
        lt_factor->apply(scalar, id, scalar, factors);
    }
}


#define GKO_DECLARE_SYMBOLIC_CHOLESKY(ValueType, IndexType)            \
    void symbolic_cholesky(                                            \
        const matrix::Csr<ValueType, IndexType>* mtx, bool symmetrize, \
        std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors,   \
        std::unique_ptr<factorization::elimination_forest<IndexType>>& forest)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SYMBOLIC_CHOLESKY);


template <typename ValueType, typename IndexType>
void symbolic_lu_near_symm(
    const matrix::Csr<ValueType, IndexType>* mtx,
    std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors)
{
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using float_matrix_type = matrix::Csr<float, IndexType>;
    using scalar_type = gko::matrix::Dense<float>;
    using id_type = gko::matrix::Identity<float>;
    GKO_ASSERT_IS_SQUARE_MATRIX(mtx);
    const auto exec = mtx->get_executor();
    const auto size = mtx->get_size();
    std::unique_ptr<float_matrix_type> symm_factors;
    {
        const auto nnz = mtx->get_num_stored_elements();
        // turn the input matrix into a symbolic float matrix
        array<float> dummy_values{exec, nnz};
        const auto float_mtx = float_matrix_type::create_const(
            exec, size, dummy_values.as_const_view(),
            make_const_array_view(exec, nnz, mtx->get_const_col_idxs()),
            make_const_array_view(exec, size[0] + 1,
                                  mtx->get_const_row_ptrs()));
        // compute A + A^T symbolically
        const auto scalar = gko::initialize<scalar_type>({one<float>()}, exec);
        const auto symm_mtx = as<float_matrix_type>(float_mtx->transpose());
        const auto id = id_type::create(exec, size);
        float_mtx->apply(scalar, id, scalar, symm_mtx);
        // compute Cholesky factorization
        std::unique_ptr<elimination_forest<IndexType>> forest;
        symbolic_cholesky(symm_mtx.get(), true, symm_factors, forest);
    }
    // build lookup structure
    array<IndexType> storage_offsets{exec, size[0] + 1};
    array<int64> row_descs{exec, size[0]};
    array<IndexType> diag_idxs{exec, size[0]};
    const auto allowed_sparsity = gko::matrix::csr::sparsity_type::bitmap |
                                  gko::matrix::csr::sparsity_type::full |
                                  gko::matrix::csr::sparsity_type::hash;
    exec->run(make_build_lookup_offsets(
        symm_factors->get_const_row_ptrs(), symm_factors->get_const_col_idxs(),
        size[0], allowed_sparsity, storage_offsets.get_data()));
    const auto storage_size =
        static_cast<size_type>(get_element(storage_offsets, size[0]));
    array<int32> storage{exec, storage_size};
    exec->run(make_build_lookup(
        symm_factors->get_const_row_ptrs(), symm_factors->get_const_col_idxs(),
        size[0], allowed_sparsity, storage_offsets.get_const_data(),
        row_descs.get_data(), storage.get_data()));
    // compute "numerical" factorization with 1s and 0s
    array<IndexType> factor_row_ptrs{exec, size[0] + 1};
    exec->run(make_symbolic_factorize_simple(
        mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
        storage_offsets.get_const_data(), row_descs.get_const_data(),
        storage.get_const_data(), symm_factors.get(),
        factor_row_ptrs.get_data()));
    // build row pointers from nnz
    exec->run(
        make_prefix_sum_nonnegative(factor_row_ptrs.get_data(), size[0] + 1));
    const auto factor_nnz =
        static_cast<size_type>(get_element(factor_row_ptrs, size[0]));
    // copy over nonzero columns
    array<IndexType> factor_cols{exec, factor_nnz};
    exec->run(make_symbolic_factorize_simple_finalize(symm_factors.get(),
                                                      factor_cols.get_data()));
    factors =
        matrix_type::create(exec, size, array<ValueType>{exec, factor_nnz},
                            std::move(factor_cols), std::move(factor_row_ptrs));
}


#define GKO_DECLARE_SYMBOLIC_LU_NEAR_SYMM(ValueType, IndexType) \
    void symbolic_lu_near_symm(                                 \
        const matrix::Csr<ValueType, IndexType>* mtx,           \
        std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SYMBOLIC_LU_NEAR_SYMM);


template <typename ValueType, typename IndexType>
void symbolic_lu(const matrix::Csr<ValueType, IndexType>* mtx,
                 std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors)
{
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    const auto exec = mtx->get_executor();
    const auto host_exec = exec->get_master();
    const auto num_rows = mtx->get_size()[0];
    const auto host_mtx = make_temporary_clone(host_exec, mtx);
    const auto in_row_ptrs = host_mtx->get_const_row_ptrs();
    const auto in_col_idxs = host_mtx->get_const_col_idxs();
    array<IndexType> host_out_row_ptr_array(host_exec, num_rows + 1);
    const auto out_row_ptrs = host_out_row_ptr_array.get_data();
    vector<IndexType> fill(num_rows, host_exec);
    vector<IndexType> out_col_idxs(host_exec);
    vector<IndexType> diags(num_rows, host_exec);
    deque<IndexType> frontier_queue(host_exec);
    for (IndexType row = 0; row < num_rows; row++) {
        out_row_ptrs[row] = out_col_idxs.size();
        fill[row] = row;
        // first copy over the original row and add all lower triangular entries
        // to the queue
        const auto row_begin = in_row_ptrs[row];
        const auto row_end = in_row_ptrs[row + 1];
        for (auto nz = row_begin; nz < row_end; nz++) {
            const auto col = in_col_idxs[nz];
            fill[col] = row;
            if (col < row) {
                frontier_queue.push_back(col);
            }
            out_col_idxs.push_back(col);
        }
        // then add fill-in for all queued entries
        while (!frontier_queue.empty()) {
            const auto frontier = frontier_queue.front();
            assert(frontier < row);
            frontier_queue.pop_front();
            // add the fill-in for this node from U
            const auto upper_begin = diags[frontier] + 1;
            const auto upper_end = out_row_ptrs[frontier + 1];
            for (auto nz = upper_begin; nz < upper_end; nz++) {
                const auto col = out_col_idxs[nz];
                if (fill[col] < row) {
                    fill[col] = row;
                    out_col_idxs.push_back(col);
                    // any fill-in on the lower triangle may introduce
                    // additional fill-in when eliminated, so we need to enqueue
                    // it as well.
                    if (col < row) {
                        frontier_queue.push_back(col);
                    }
                }
            }
        }
        // restore sorting and find diagonal entry to separate L and U
        const auto row_begin_it = out_col_idxs.begin() + out_row_ptrs[row];
        const auto row_end_it = out_col_idxs.end();
        std::sort(row_begin_it, row_end_it);
        auto row_diag_it = std::lower_bound(row_begin_it, row_end_it, row);
        // add diagonal if it's missing
        if (row_diag_it == row_end_it || *row_diag_it != row) {
            row_diag_it = out_col_idxs.insert(row_diag_it, row);
        }
        diags[row] = std::distance(out_col_idxs.begin(), row_diag_it);
    }
    const auto out_nnz = static_cast<size_type>(out_col_idxs.size());
    out_row_ptrs[num_rows] = out_nnz;
    array<IndexType> out_row_ptr_array{exec, std::move(host_out_row_ptr_array)};
    array<IndexType> out_col_idx_array{exec, out_nnz};
    array<ValueType> out_val_array{exec, out_nnz};
    exec->copy_from(host_exec, out_nnz, out_col_idxs.data(),
                    out_col_idx_array.get_data());
    factors = matrix_type::create(
        exec, mtx->get_size(), std::move(out_val_array),
        std::move(out_col_idx_array), std::move(out_row_ptr_array));
}


#define GKO_DECLARE_SYMBOLIC_LU(ValueType, IndexType) \
    void symbolic_lu(                                 \
        const matrix::Csr<ValueType, IndexType>* mtx, \
        std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SYMBOLIC_LU);


}  // namespace factorization
}  // namespace gko

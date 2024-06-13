// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/cholesky_kernels.hpp"


#include <algorithm>
#include <memory>


#include <CL/sycl.hpp>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/elimination_forest.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Cholesky namespace.
 *
 * @ingroup factor
 */
namespace cholesky {


template <typename ValueType, typename IndexType>
void symbolic_count(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* mtx,
                    const factorization::elimination_forest<IndexType>& forest,
                    IndexType* row_nnz, array<IndexType>& tmp_storage)
{
    const auto num_rows = mtx->get_size()[0];
    const auto mtx_nnz = mtx->get_num_stored_elements();
    tmp_storage.resize_and_reset(mtx_nnz + num_rows);
    const auto postorder_cols = tmp_storage.get_data();
    const auto lower_ends = postorder_cols + mtx_nnz;
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto cols = mtx->get_const_col_idxs();
    const auto inv_postorder = forest.inv_postorder.get_const_data();
    const auto postorder_parent = forest.postorder_parents.get_const_data();
    auto queue = exec->get_queue();
    // build sorted postorder node list for each row
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx_id) {
            const auto row = idx_id[0];
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            auto lower_end = row_begin;
            for (auto nz = row_begin; nz < row_end; nz++) {
                const auto col = cols[nz];
                if (col < row) {
                    postorder_cols[lower_end] = inv_postorder[cols[nz]];
                    lower_end++;
                }
            }
            // heap-sort the elements
            std::make_heap(postorder_cols + row_begin,
                           postorder_cols + lower_end);
            std::sort_heap(postorder_cols + row_begin,
                           postorder_cols + lower_end);
            lower_ends[row] = lower_end;
        });
    });
    // count nonzeros per row of L
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx_id) {
            const auto row = idx_id[0];
            const auto row_begin = row_ptrs[row];
            // instead of relying on the input containing a diagonal, we
            // artificially introduce the diagonal entry (in postorder indexing)
            // as a sentinel after the last lower triangular entry.
            const auto diag_postorder = inv_postorder[row];
            const auto lower_end = lower_ends[row];
            IndexType count{};
            for (auto nz = row_begin; nz < lower_end; ++nz) {
                auto node = postorder_cols[nz];
                const auto next_node = nz < lower_end - 1
                                           ? postorder_cols[nz + 1]
                                           : diag_postorder;
                while (node < next_node) {
                    count++;
                    node = postorder_parent[node];
                }
            }
            row_nnz[row] = count + 1;  // lower entries plus diagonal
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT);


template <typename ValueType, typename IndexType>
void symbolic_factorize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* mtx,
    const factorization::elimination_forest<IndexType>& forest,
    matrix::Csr<ValueType, IndexType>* l_factor,
    const array<IndexType>& tmp_storage)
{
    const auto num_rows = mtx->get_size()[0];
    const auto mtx_nnz = mtx->get_num_stored_elements();
    const auto postorder_cols = tmp_storage.get_const_data();
    const auto lower_ends = postorder_cols + mtx_nnz;
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto postorder = forest.postorder.get_const_data();
    const auto inv_postorder = forest.inv_postorder.get_const_data();
    const auto postorder_parent = forest.postorder_parents.get_const_data();
    const auto out_row_ptrs = l_factor->get_const_row_ptrs();
    const auto out_cols = l_factor->get_col_idxs();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx_id) {
            const auto row = idx_id[0];
            const auto row_begin = row_ptrs[row];
            // instead of relying on the input containing a diagonal, we
            // artificially introduce the diagonal entry (in postorder indexing)
            // as a sentinel after the last lower triangular entry.
            const auto diag_postorder = inv_postorder[row];
            const auto lower_end = lower_ends[row];
            auto out_nz = out_row_ptrs[row];
            for (auto nz = row_begin; nz < lower_end; ++nz) {
                auto node = postorder_cols[nz];
                const auto next_node = nz < lower_end - 1
                                           ? postorder_cols[nz + 1]
                                           : diag_postorder;
                while (node < next_node) {
                    out_cols[out_nz] = postorder[node];
                    out_nz++;
                    node = postorder_parent[node];
                }
            }
            // add diagonal entry
            out_cols[out_nz] = row;
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE);


template <typename ValueType, typename IndexType>
void forest_from_factor(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* factors,
                        gko::factorization::elimination_forest<IndexType>&
                            forest) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_FOREST_FROM_FACTOR);


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                IndexType* transpose_idxs,
                matrix::Csr<ValueType, IndexType>* factors) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_INITIALIZE);


template <typename ValueType, typename IndexType>
void factorize(std::shared_ptr<const DefaultExecutor> exec,
               const IndexType* lookup_offsets, const int64* lookup_descs,
               const int32* lookup_storage, const IndexType* diag_idxs,
               const IndexType* transpose_idxs,
               const factorization::elimination_forest<IndexType>& forest,
               matrix::Csr<ValueType, IndexType>* factors,
               array<int>& tmp_storage) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_FACTORIZE);


}  // namespace cholesky
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko

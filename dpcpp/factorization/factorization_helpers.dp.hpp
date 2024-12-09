// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sycl/sycl.hpp>

#include "core/factorization/factorization_helpers.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace factorization {
namespace helpers {

using namespace ::gko::factorization;


template <typename ValueType, typename IndexType, typename LClosure,
          typename UClosure>
void initialize_l_u(size_type num_rows, const IndexType* __restrict__ row_ptrs,
                    const IndexType* __restrict__ col_idxs,
                    const ValueType* __restrict__ values,
                    const IndexType* __restrict__ l_row_ptrs,
                    IndexType* __restrict__ l_col_idxs,
                    ValueType* __restrict__ l_values,
                    const IndexType* __restrict__ u_row_ptrs,
                    IndexType* __restrict__ u_col_idxs,
                    ValueType* __restrict__ u_values, LClosure l_closure,
                    UClosure u_closure, sycl::nd_item<3> item_ct1)
{
    const auto row = thread::get_thread_id_flat<IndexType>(item_ct1);
    if (row < num_rows) {
        auto l_idx = l_row_ptrs[row];
        auto u_idx = u_row_ptrs[row] + 1;  // we treat the diagonal separately
        // default diagonal to one
        auto diag_val = one<ValueType>();
        for (size_type i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto col = col_idxs[i];
            const auto val = values[i];
            // save diagonal entry for later
            if (col == row) {
                diag_val = val;
            }
            if (col < row) {
                l_col_idxs[l_idx] = col;
                l_values[l_idx] = l_closure.map_off_diag(val);
                ++l_idx;
            }
            if (row < col) {
                u_col_idxs[u_idx] = col;
                u_values[u_idx] = u_closure.map_off_diag(val);
                ++u_idx;
            }
        }
        // store diagonal entries
        auto l_diag_idx = l_row_ptrs[row + 1] - 1;
        auto u_diag_idx = u_row_ptrs[row];
        l_col_idxs[l_diag_idx] = row;
        u_col_idxs[u_diag_idx] = row;
        l_values[l_diag_idx] = l_closure.map_diag(diag_val);
        u_values[u_diag_idx] = u_closure.map_diag(diag_val);
    }
}


template <typename ValueType, typename IndexType, typename LClosure>
void initialize_l(size_type num_rows, const IndexType* __restrict__ row_ptrs,
                  const IndexType* __restrict__ col_idxs,
                  const ValueType* __restrict__ values,
                  const IndexType* __restrict__ l_row_ptrs,
                  IndexType* __restrict__ l_col_idxs,
                  ValueType* __restrict__ l_values, LClosure l_closure,
                  sycl::nd_item<3> item_ct1)
{
    const auto row = thread::get_thread_id_flat<IndexType>(item_ct1);
    if (row < num_rows) {
        auto l_idx = l_row_ptrs[row];
        // if there was no diagonal entry, default to one
        auto diag_val = one<ValueType>();
        for (size_type i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto col = col_idxs[i];
            const auto val = values[i];
            // save diagonal entry for later
            if (col == row) {
                diag_val = val;
            }
            if (col < row) {
                l_col_idxs[l_idx] = col;
                l_values[l_idx] = l_closure.map_off_diag(val);
                ++l_idx;
            }
        }
        // store diagonal entries
        auto l_diag_idx = l_row_ptrs[row + 1] - 1;
        l_col_idxs[l_diag_idx] = row;
        l_values[l_diag_idx] = l_closure.map_diag(diag_val);
    }
}


}  // namespace helpers
}  // namespace factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko

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
#ifndef GKO_DPCPP_MATRIX_BATCH_ELL_KERNELS_HPP_
#define GKO_DPCPP_MATRIX_BATCH_ELL_KERNELS_HPP_

#include "core/matrix/batch_ell_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/segment_scan.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_ell
 */
namespace batch_ell {

template <typename ValueType>
inline void matvec_kernel(
    sycl::nd_item<3> item_ct1,
    const gko::batch_ell::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int tidx = item_ct1.get_local_linear_id(); tidx < a.num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < a.num_stored_elems_per_row; idx++) {
            const auto col_idx = a.col_idxs[tidx + idx * a.stride];
            if (col_idx < idx)
                break;
            else
                temp += a.values[tidx + idx * a.stride] *
                        b.values[col_idx * b.stride];
        }
        c.values[tidx * c.stride] = temp;
    }
}


template <typename ValueType>
inline void advanced_matvec_kernel(
    sycl::nd_item<3> item_ct1, const ValueType alpha,
    const gko::batch_ell::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int tidx = item_ct1.get_local_linear_id(); tidx < a.num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < a.num_stored_elems_per_row; idx++) {
            const auto col_idx = a.col_idxs[tidx + idx * a.stride];
            if (col_idx < idx)
                break;
            else
                temp += alpha * a.values[tidx + idx * a.stride] *
                        b.values[col_idx * b.stride];
        }
        c.values[tidx * c.stride] = temp + beta * c.values[tidx * c.stride];
    }
}


template <typename IndexType>
inline void check_diagonal_entries_kernel(
    sycl::nd_item<3> item_ct1, const IndexType num_min_rows_cols,
    const size_type row_stride, const size_type max_nnz_per_row,
    const IndexType* const __restrict__ col_idxs,
    bool* const __restrict__ has_all_diags)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    if (item_ct1.get_local_linear_id() == 0) {
        *has_all_diags = true;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (sg_id == 0 && num_min_rows_cols > 0) {
        bool row_has_diag_local{false};
        if (sg.get_local_id() == 0) {
            if (col_idxs[0] == 0) {
                row_has_diag_local = true;
            }
        }
        auto row_has_diag =
            sycl::ext::oneapi::group_ballot(sg, row_has_diag_local).any();
        if (!row_has_diag) {
            if (sg.get_local_id() == 0) {
                *has_all_diags = false;
            }
            return;
        }
    } else if (sg_id < num_min_rows_cols) {
        bool row_has_diag_local{false};
        for (IndexType iz = sg.get_local_id(); iz < max_nnz_per_row;
             iz += sg_size) {
            if (col_idxs[iz * row_stride + sg_id] == sg_id) {  // or = sg_id
                row_has_diag_local = true;
                break;
            }
        }
        auto row_has_diag =
            sycl::ext::oneapi::group_ballot(sg, row_has_diag_local).any();
        if (!row_has_diag) {
            if (sg.get_local_id() == 0) {
                *has_all_diags = false;
            }
            return;
        }
    }
}


template <typename ValueType>
inline void add_scaled_identity_kernel(
    sycl::nd_item<3> item_ct1, const int nrows, const size_type row_stride,
    const int max_nnz_per_row, const int* const col_idxs,
    ValueType* const __restrict__ values, const ValueType& alpha,
    const ValueType& beta)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < nrows; row += num_sg) {
        if (row == 0) {
            for (int iz = sg.get_local_id(); iz < max_nnz_per_row;
                 iz += sg_size) {
                values[iz * row_stride] *= beta;
            }
            if (sg.get_local_id() == 0 && col_idxs[0] == 0) {
                values[0] += alpha;
            }
        } else {
            for (int iz = sg.get_local_id(); iz < max_nnz_per_row;
                 iz += sg_size) {
                values[iz * row_stride + row] *= beta;
                if (row == col_idxs[iz * row_stride + row]) {
                    values[iz * row_stride + row] += alpha;
                }
            }
        }
    }
}


}  // namespace batch_ell
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif

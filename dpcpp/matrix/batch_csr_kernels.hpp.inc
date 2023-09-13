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

template <typename ValueType>
__dpct_inline__ void matvec_kernel(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const gko::batch_dense::BatchEntry<ValueType>& c,
    sycl::nd_item<3>& item_ct1)
{
    auto sg = item_ct1.get_sub_group();

    for (int row_and_rhs_combination = sg.get_group_id();
         row_and_rhs_combination < a.num_rows * b.num_rhs;
         row_and_rhs_combination += sg.get_group_range().size()) {
        const int row = row_and_rhs_combination / b.num_rhs;
        const int rhs = row_and_rhs_combination % b.num_rhs;

        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = sg.get_local_id() + row_start; i < row_end;
             i += sg.get_local_range().size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];
            temp += val * b.values[col * b.stride + rhs];
        }

        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());

        if (sg.get_local_id() == 0) {
            c.values[row * c.stride + rhs] = temp;
        }
    }
}


template <typename ValueType>
__dpct_inline__ void single_matvec_kernel(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c,
    sycl::nd_item<3>& item_ct1)
{
    const auto num_rows = a.num_rows;
    const auto val = a.values;
    const auto col = a.col_idxs;
    for (int tidx = item_ct1.get_local_linear_id(); tidx < num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (auto nnz = a.row_ptrs[tidx]; nnz < a.row_ptrs[tidx + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += val[nnz] * b[col_idx];
        }
        c[tidx] = temp;
    }
}


template <typename ValueType>
__dpct_inline__ void advanced_matvec_kernel(
    const ValueType alpha, const gko::batch_csr::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& c,
    sycl::nd_item<3>& item_ct1)
{
    auto sg = item_ct1.get_sub_group();

    for (int row_and_rhs_combination = sg.get_group_id();
         row_and_rhs_combination < a.num_rows * b.num_rhs;
         row_and_rhs_combination += sg.get_group_range().size()) {
        const int row = row_and_rhs_combination / b.num_rhs;
        const int rhs = row_and_rhs_combination % b.num_rhs;

        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = sg.get_local_id() + row_start; i < row_end;
             i += sg.get_local_range().size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];
            temp += alpha * val * b.values[col * b.stride + rhs];
        }

        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());

        if (sg.get_local_id() == 0) {
            c.values[row * c.stride + rhs] =
                temp + beta * c.values[row * c.stride + rhs];
        }
    }
}


template <typename ValueType>
__dpct_inline__ void single_advanced_matvec_kernel(
    const ValueType alpha, const gko::batch_csr::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c, sycl::nd_item<3>& item_ct1)
{
    auto sg = item_ct1.get_sub_group();

    for (int row = sg.get_group_id(); row < a.num_rows;
         row += sg.get_group_range().size()) {
        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = sg.get_local_id() + row_start; i < row_end;
             i += sg.get_local_range().size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];
            temp += val * b[col];
        }

        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());

        if (sg.get_local_id() == 0) {
            c[row] = alpha * temp + beta * c[row];
        }
    }
}


template <typename ValueType>
void batch_scale_kernel(  // TODO: consider to find a new kernel name
    const ValueType* const left_scale, const ValueType* const right_scale,
    const gko::batch_csr::BatchEntry<ValueType>& a, sycl::nd_item<3>& item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int i_row = sg_id; i_row < a.num_rows; i_row += num_sg) {
        const ValueType rowscale = left_scale[i_row];
        for (int iz = a.row_ptrs[i_row] + sg.get_local_id();
             iz < a.row_ptrs[i_row + 1]; iz += sg_size) {
            a.values[iz] *= rowscale * right_scale[a.col_idxs[iz]];
        }
    }
}


template <typename ValueType>
__dpct_inline__ void pre_diag_scale_kernel(
    const int num_rows, ValueType* const __restrict__ a_values,
    const int* const __restrict__ col_idxs,
    const int* const __restrict__ row_ptrs, const int num_rhs,
    const size_type b_stride, ValueType* const __restrict__ b,
    const ValueType* const __restrict__ left_scale,
    const ValueType* const __restrict__ right_scale, sycl::nd_item<3>& item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_max_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int i_row = sg_id; i_row < num_rows; i_row += num_sg) {
        const ValueType rowscale = left_scale[i_row];
        for (int iz = row_ptrs[i_row] + sg.get_local_id();
             iz < row_ptrs[i_row + 1]; iz += sg_size) {
            a_values[iz] *= rowscale * right_scale[col_idxs[iz]];
        }
    }
    for (int iz = item_ct1.get_local_linear_id(); iz < num_rows * num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        b[row * b_stride + col] *= left_scale[row];
    }
}


template <typename ValueType>
__dpct_inline__ void convert_to_batch_dense_kernel(
    const int num_rows, const int num_cols, const int* const row_ptrs,
    const int* const col_idxs, const ValueType* const values,
    const size_type dense_stride, ValueType* const dense,
    sycl::nd_item<3>& item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int i_row = sg_id; i_row < num_rows; i_row += num_sg) {
        for (int j = sg.get_local_id(); j < num_cols; j += sg_size) {
            dense[i_row * dense_stride + j] = zero<ValueType>();
        }
        for (int iz = row_ptrs[i_row] + sg.get_local_id();
             iz < row_ptrs[i_row + 1]; iz += sg_size) {
            dense[i_row * dense_stride + col_idxs[iz]] = values[iz];
        }
    }
}


__dpct_inline__ void check_all_diagonal_kernel(
    const int min_rows_cols, const int* const __restrict__ row_ptrs,
    const int* const __restrict__ col_idxs, bool* const __restrict__ all_diags,
    int* tile_has_diags, sycl::nd_item<3>& item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    int this_tile_has_diags = 1;
    for (int row = sg_id; row < min_rows_cols; row += num_sg) {
        const int row_sz = row_ptrs[row + 1] - row_ptrs[row];
        int has_diag = 0;
        for (int iz = sg.get_local_id(); iz < row_sz; iz += sg_size) {
            has_diag = static_cast<int>(col_idxs[iz + row_ptrs[row]] == row);
            if (has_diag) {
                break;
            }
        }
        auto row_has_diag = sycl::ext::oneapi::group_ballot(sg, has_diag).any();
        this_tile_has_diags = this_tile_has_diags && row_has_diag;
    }
    if (sg.get_local_id() == 0) {
        tile_has_diags[sg_id] = this_tile_has_diags;
    }

    // workgroup sync, must-have
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // reduce array to one warp
    if (sg_id == 0) {
        for (int i = sg_size + sg.get_local_id(); i < num_sg; i += sg_size) {
            tile_has_diags[i % sg_size] =
                tile_has_diags[i % sg_size] && tile_has_diags[i];
        }
        // warp-reduce
        int var =
            sg.get_local_id() < num_sg ? tile_has_diags[sg.get_local_id()] : 1;
        var = sycl::ext::oneapi::group_ballot(sg, var).all();
        if (sg.get_local_id() == 0) {
            all_diags[0] = static_cast<bool>(var);
        }
    }
}


template <typename ValueType>
__dpct_inline__ void add_scaled_identity_kernel(
    const int num_rows, const int* const row_ptrs, const int* const col_idxs,
    ValueType* const __restrict__ values, const ValueType& alpha,
    const ValueType& beta, sycl::nd_item<3>& item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < num_rows; row += num_sg) {
        for (int iz = row_ptrs[row] + sg.get_local_id(); iz < row_ptrs[row + 1];
             iz += sg_size) {
            values[iz] *= beta;
            if (row == col_idxs[iz]) {
                values[iz] += alpha;
            }
        }
    }
}

/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

namespace kernel {


/**
 * The global function for counting the number of nonzeros per row of COO.
 * It is almost like COO spmv routine.
 * It performs is_nonzeros(Coo) times the vector whose values are one
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_line  the maximum round of each warp
 * @param val  the value array of the matrix
 * @param row  the row index array of the matrix
 * @param nnz_per_row  the output nonzeros per row
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_coo_row_nnz(
    const size_type nnz, const size_type num_lines,
    const ValueType* __restrict__ val, const IndexType* __restrict__ row,
    IndexType* __restrict__ nnz_per_row)
{
    IndexType temp_val = 0;
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    size_type num = (nnz > start) * ceildiv(nnz - start, subwarp_size);
    num = min(num, num_lines);
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 1) * subwarp_size;
    IndexType ind = ind_start;
    IndexType curr_row = (ind < nnz) ? row[ind] : 0;
    const auto tile_block =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    for (; ind < ind_end; ind += subwarp_size) {
        temp_val += ind < nnz && val[ind] != zero<ValueType>();
        auto next_row =
            (ind + subwarp_size < nnz) ? row[ind + subwarp_size] : row[nnz - 1];
        // segmented scan
        if (tile_block.any(curr_row != next_row)) {
            bool is_first_in_segment =
                segment_scan<subwarp_size>(tile_block, curr_row, &temp_val);
            if (is_first_in_segment) {
                atomic_add(&(nnz_per_row[curr_row]), temp_val);
            }
            temp_val = 0;
        }
        curr_row = next_row;
    }
    if (num > 0) {
        ind = ind_end;
        temp_val += ind < nnz && val[ind] != zero<ValueType>();
        // segmented scan

        bool is_first_in_segment =
            segment_scan<subwarp_size>(tile_block, curr_row, &temp_val);
        if (is_first_in_segment) {
            atomic_add(&(nnz_per_row[curr_row]), temp_val);
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_csr(
    size_type num_rows, size_type max_nnz_per_row, size_type stride,
    const ValueType* __restrict__ ell_val,
    const IndexType* __restrict__ ell_col,
    const ValueType* __restrict__ coo_val,
    const IndexType* __restrict__ coo_col,
    const IndexType* __restrict__ coo_offset,
    IndexType* __restrict__ result_row_ptrs,
    IndexType* __restrict__ result_col_idxs,
    ValueType* __restrict__ result_values)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx < num_rows) {
        auto write_to = result_row_ptrs[tidx];
        for (size_type i = 0; i < max_nnz_per_row; i++) {
            const auto source_idx = tidx + stride * i;
            if (ell_val[source_idx] != zero<ValueType>()) {
                result_values[write_to] = ell_val[source_idx];
                result_col_idxs[write_to] = ell_col[source_idx];
                write_to++;
            }
        }
        for (auto i = coo_offset[tidx]; i < coo_offset[tidx + 1]; i++) {
            if (coo_val[i] != zero<ValueType>()) {
                result_values[write_to] = coo_val[i];
                result_col_idxs[write_to] = coo_col[i];
                write_to++;
            }
        }
    }
}


template <typename ValueType1, typename ValueType2>
__global__ __launch_bounds__(default_block_size) void add(
    size_type num, ValueType1* __restrict__ val1,
    const ValueType2* __restrict__ val2)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num) {
        val1[tidx] += val2[tidx];
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
struct hybrid_tuple_unpack_functor {
    using device_entry = device_type<matrix_data_entry<ValueType, IndexType>>;
    device_entry __device__
    operator()(thrust::tuple<size_type, device_entry> e) const
    {
        return thrust::get<1>(e);
    }
};


template <typename ValueType, typename IndexType>
void split_matrix_data(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, IndexType>>& data,
    const int64* row_ptrs, size_type ell_limit, size_type num_rows,
    Array<matrix_data_entry<ValueType, IndexType>>& ell_data,
    Array<matrix_data_entry<ValueType, IndexType>>& coo_data)
{
    using device_entry = device_type<matrix_data_entry<ValueType, IndexType>>;
    auto iota = thrust::make_counting_iterator(size_type{});
    auto data_it =
        thrust::device_pointer_cast(as_device_type(data.get_const_data()));
    const auto nnz = data.get_num_elems();
    auto enumerated_data_it =
        thrust::make_zip_iterator(thrust::make_tuple(iota, data_it));
    auto ell_predicate = [row_ptrs, ell_limit] __device__(
                             thrust::tuple<size_type, device_entry> e) {
        const auto row_begin = row_ptrs[thrust::get<1>(e).row];
        const auto local_nz = thrust::get<0>(e) - row_begin;
        return local_nz < ell_limit;
    };
    auto coo_predicate = [row_ptrs, ell_limit] __device__(
                             thrust::tuple<size_type, device_entry> e) {
        const auto row_begin = row_ptrs[thrust::get<1>(e).row];
        const auto local_nz = thrust::get<0>(e) - row_begin;
        return local_nz >= ell_limit;
    };
    const auto ell_nnz = static_cast<size_type>(
        thrust::count_if(thrust::device, enumerated_data_it,
                         enumerated_data_it + nnz, ell_predicate));
    const auto coo_nnz = nnz - ell_nnz;
    ell_data.resize_and_reset(ell_nnz);
    coo_data.resize_and_reset(coo_nnz);
    auto ell_data_it = thrust::make_transform_output_iterator(
        thrust::device_pointer_cast(as_device_type(ell_data.get_data())),
        hybrid_tuple_unpack_functor<ValueType, IndexType>{});
    auto coo_data_it = thrust::make_transform_output_iterator(
        thrust::device_pointer_cast(as_device_type(coo_data.get_data())),
        hybrid_tuple_unpack_functor<ValueType, IndexType>{});
    thrust::copy_if(thrust::device, enumerated_data_it,
                    enumerated_data_it + nnz, ell_data_it, ell_predicate);
    thrust::copy_if(thrust::device, enumerated_data_it,
                    enumerated_data_it + nnz, coo_data_it, coo_predicate);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_SPLIT_MATRIX_DATA_KERNEL);

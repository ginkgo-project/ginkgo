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

template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  array<ValueType>& values, array<IndexType>& row_idxs,
                  array<IndexType>& col_idxs)
{
    // workaround for CUDA 9.2 Thrust: Their complex<> implementation is broken
    // due to overly generic assignment operator and constructor leading to
    // ambiguities. So we need to use our own fake_complex type
    using device_value_type = device_member_type<ValueType>;
    auto value_ptr = thrust::device_pointer_cast(
        reinterpret_cast<const device_value_type*>(values.get_const_data()));
    auto size = values.get_num_elems();
    // count nonzeros
    auto nnz =
        thrust::count_if(thrust::device, value_ptr, value_ptr + size,
                         [] __device__(device_value_type value) {
                             return is_nonzero(fake_complex_unpack(value));
                         });
    if (nnz < size) {
        using tuple_type =
            thrust::tuple<IndexType, IndexType, device_value_type>;
        // allocate new storage
        array<ValueType> new_values{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_row_idxs{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_col_idxs{exec, static_cast<size_type>(nnz)};
        // copy nonzeros
        auto it = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(row_idxs.get_const_data()),
            thrust::device_pointer_cast(col_idxs.get_const_data()), value_ptr));
        auto out_it = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(new_row_idxs.get_data()),
            thrust::device_pointer_cast(new_col_idxs.get_data()),
            thrust::device_pointer_cast(
                reinterpret_cast<device_value_type*>(new_values.get_data()))));
        thrust::copy_if(
            thrust::device, it, it + size, out_it,
            [] __device__(tuple_type entry) {
                return is_nonzero(fake_complex_unpack(thrust::get<2>(entry)));
            });
        // swap out storage
        values = std::move(new_values);
        row_idxs = std::move(new_row_idxs);
        col_idxs = std::move(new_col_idxs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void sum_duplicates(std::shared_ptr<const DefaultExecutor> exec, size_type,
                    array<ValueType>& values, array<IndexType>& row_idxs,
                    array<IndexType>& col_idxs)
{
    using device_value_type = device_member_type<ValueType>;
    const auto size = values.get_num_elems();
    const auto rows = row_idxs.get_const_data();
    const auto cols = col_idxs.get_const_data();
    auto iota = thrust::make_counting_iterator(size_type{});
    const auto new_size = static_cast<size_type>(thrust::count_if(
        thrust::device, iota, iota + size,
        [rows, cols] __device__(size_type i) {
            const auto prev_row =
                i > 0 ? rows[i - 1] : invalid_index<IndexType>();
            const auto prev_col =
                i > 0 ? cols[i - 1] : invalid_index<IndexType>();
            return rows[i] != prev_row || cols[i] != prev_col;
        }));
    if (new_size < size) {
        // allocate new storage
        array<ValueType> new_values{exec, new_size};
        array<IndexType> new_row_idxs{exec, new_size};
        array<IndexType> new_col_idxs{exec, new_size};
        // reduce duplicates
        auto in_locs = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_pointer_cast(rows),
                               thrust::device_pointer_cast(cols)));
        auto in_vals = thrust::device_pointer_cast(
            reinterpret_cast<const device_value_type*>(
                values.get_const_data()));
        auto out_locs = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(new_row_idxs.get_data()),
            thrust::device_pointer_cast(new_col_idxs.get_data())));
        auto out_vals = thrust::device_pointer_cast(
            reinterpret_cast<device_value_type*>(new_values.get_data()));
        thrust::reduce_by_key(thrust::device, in_locs, in_locs + size, in_vals,
                              out_locs, out_vals);
        // swap out storage
        values = std::move(new_values);
        row_idxs = std::move(new_row_idxs);
        col_idxs = std::move(new_col_idxs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SUM_DUPLICATES_KERNEL);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,
                    device_matrix_data<ValueType, IndexType>& data)
{
    // workaround for CUDA 9.2 Thrust: Their complex<> implementation is broken
    // due to overly generic assignment operator and constructor leading to
    // ambiguities. So we need to use our own fake_complex type
    using device_value_type = device_member_type<ValueType>;
    auto it = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::device_pointer_cast(data.get_row_idxs()),
                           thrust::device_pointer_cast(data.get_col_idxs())));
    auto vals = thrust::device_pointer_cast(
        reinterpret_cast<device_value_type*>(data.get_values()));
    thrust::sort_by_key(thrust::device, it, it + data.get_num_elems(), vals);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);

template <typename ValueType, typename SortingIndexType, typename IndexType>
void sort_row_major_with_scatter(std::shared_ptr<const DefaultExecutor> exec,
                                 device_matrix_data<ValueType, IndexType>& data,
                                 array<SortingIndexType>& scatter_pattern)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL_WITH_SCATTER);

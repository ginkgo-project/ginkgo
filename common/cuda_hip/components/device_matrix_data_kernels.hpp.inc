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

template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  Array<ValueType>& values, Array<IndexType>& row_idxs,
                  Array<IndexType>& col_idxs)
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
        Array<ValueType> new_values{exec, static_cast<size_type>(nnz)};
        Array<IndexType> new_row_idxs{exec, static_cast<size_type>(nnz)};
        Array<IndexType> new_col_idxs{exec, static_cast<size_type>(nnz)};
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

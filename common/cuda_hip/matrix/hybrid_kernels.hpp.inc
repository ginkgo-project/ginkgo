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


namespace {


template <typename ValueType, typename IndexType>
struct hybrid_tuple_unpack_functor {
    using device_entry = device_type<matrix_data_entry<ValueType, IndexType>>;
    device_entry __device__
    operator()(thrust::tuple<size_type, device_entry> e) const
    {
        return thrust::get<1>(e);
    }
};


}  // anonymous namespace


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

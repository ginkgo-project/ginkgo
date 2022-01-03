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

#include <ginkgo/core/base/device_matrix_data.hpp>


#include <ginkgo/core/base/executor.hpp>


#include "core/components/device_matrix_data_kernels.hpp"


namespace gko {
namespace components {
namespace {


GKO_REGISTER_OPERATION(remove_zeros, components::remove_zeros);
GKO_REGISTER_OPERATION(sort_row_major, components::sort_row_major);


}  // anonymous namespace
}  // namespace components


template <typename ValueType, typename IndexType>
device_matrix_data<ValueType, IndexType>::device_matrix_data(
    std::shared_ptr<const Executor> exec, dim<2> size, size_type nnz)
    : size{size}, nonzeros{exec, nnz}
{}


template <typename ValueType, typename IndexType>
device_matrix_data<ValueType, IndexType>::device_matrix_data(
    dim<2> size, Array<nonzero_type> data)
    : size{size}, nonzeros{std::move(data)}
{}


template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType>
device_matrix_data<ValueType, IndexType>::copy_to_host() const
{
    const auto nnz = nonzeros.get_num_elems();
    matrix_data<ValueType, IndexType> result{size};
    result.nonzeros.resize(nnz);
    nonzeros.get_executor()->get_master()->copy_from(
        nonzeros.get_executor().get(), nnz, nonzeros.get_const_data(),
        result.nonzeros.data());
    return result;
}


template <typename ValueType, typename IndexType>
device_matrix_data<ValueType, IndexType>
device_matrix_data<ValueType, IndexType>::create_view_from_host(
    std::shared_ptr<const Executor> exec, host_type& data)
{
    auto host_view = Array<nonzero_type>::view(
        exec->get_master(), data.nonzeros.size(), data.nonzeros.data());
    auto device_view = Array<nonzero_type>{exec, std::move(host_view)};
    return device_matrix_data{data.size, std::move(device_view)};
}


template <typename ValueType, typename IndexType>
void device_matrix_data<ValueType, IndexType>::remove_zeros()
{
    this->nonzeros.get_executor()->run(
        components::make_remove_zeros(this->nonzeros));
}


template <typename ValueType, typename IndexType>
void device_matrix_data<ValueType, IndexType>::sort_row_major()
{
    this->nonzeros.get_executor()->run(
        components::make_sort_row_major(this->nonzeros));
}


#define GKO_DECLARE_DEVICE_MATRIX_DATA(ValueType, IndexType) \
    struct device_matrix_data<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DEVICE_MATRIX_DATA);


}  // namespace gko

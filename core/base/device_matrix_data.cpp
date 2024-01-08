// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/device_matrix_data.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>


#include "core/base/device_matrix_data_kernels.hpp"


namespace gko {
namespace components {
namespace {


GKO_REGISTER_OPERATION(aos_to_soa, components::aos_to_soa);
GKO_REGISTER_OPERATION(soa_to_aos, components::soa_to_aos);
GKO_REGISTER_OPERATION(remove_zeros, components::remove_zeros);
GKO_REGISTER_OPERATION(sum_duplicates, components::sum_duplicates);
GKO_REGISTER_OPERATION(sort_row_major, components::sort_row_major);


}  // anonymous namespace
}  // namespace components


template <typename ValueType, typename IndexType>
device_matrix_data<ValueType, IndexType>::device_matrix_data(
    std::shared_ptr<const Executor> exec, dim<2> size, size_type num_entries)
    : size_{size},
      row_idxs_{exec, num_entries},
      col_idxs_{exec, num_entries},
      values_{exec, num_entries}
{}


template <typename ValueType, typename IndexType>
device_matrix_data<ValueType, IndexType>::device_matrix_data(
    std::shared_ptr<const Executor> exec, const device_matrix_data& data)
    : size_{data.size_},
      row_idxs_{exec, data.row_idxs_},
      col_idxs_{exec, data.col_idxs_},
      values_{exec, data.values_}
{}


template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType>
device_matrix_data<ValueType, IndexType>::copy_to_host() const
{
    const auto exec = values_.get_executor();
    const auto nnz = this->get_num_stored_elements();
    matrix_data<ValueType, IndexType> result{this->get_size()};
    result.nonzeros.resize(nnz);
    auto host_view =
        make_array_view(exec->get_master(), nnz, result.nonzeros.data());
    exec->run(components::make_soa_to_aos(
        *this, *make_temporary_clone(exec, &host_view)));
    return result;
}


template <typename ValueType, typename IndexType>
device_matrix_data<ValueType, IndexType>
device_matrix_data<ValueType, IndexType>::create_from_host(
    std::shared_ptr<const Executor> exec, const host_type& data)
{
    const auto host_view =
        make_array_view(exec->get_master(), data.nonzeros.size(),
                        const_cast<nonzero_type*>(data.nonzeros.data()));
    device_matrix_data result{exec, data.size, data.nonzeros.size()};
    exec->run(components::make_aos_to_soa(
        *make_temporary_clone(exec, &host_view), result));
    return result;
}


template <typename ValueType, typename IndexType>
void device_matrix_data<ValueType, IndexType>::sort_row_major()
{
    this->values_.get_executor()->run(components::make_sort_row_major(*this));
}


template <typename ValueType, typename IndexType>
void device_matrix_data<ValueType, IndexType>::remove_zeros()
{
    this->values_.get_executor()->run(components::make_remove_zeros(
        this->values_, this->row_idxs_, this->col_idxs_));
}


template <typename ValueType, typename IndexType>
void device_matrix_data<ValueType, IndexType>::sum_duplicates()
{
    this->sort_row_major();
    this->values_.get_executor()->run(components::make_sum_duplicates(
        this->size_[0], this->values_, this->row_idxs_, this->col_idxs_));
}


template <typename ValueType, typename IndexType>
void device_matrix_data<ValueType, IndexType>::resize_and_reset(
    size_type new_num_entries)
{
    row_idxs_.resize_and_reset(new_num_entries);
    col_idxs_.resize_and_reset(new_num_entries);
    values_.resize_and_reset(new_num_entries);
}

template <typename ValueType, typename IndexType>
void device_matrix_data<ValueType, IndexType>::resize_and_reset(
    dim<2> new_size, size_type new_num_entries)
{
    size_ = new_size;
    resize_and_reset(new_num_entries);
}


template <typename ValueType, typename IndexType>
typename device_matrix_data<ValueType, IndexType>::arrays
device_matrix_data<ValueType, IndexType>::empty_out()
{
    arrays result{std::move(row_idxs_), std::move(col_idxs_),
                  std::move(values_)};
    size_ = {};
    return result;
}


#define GKO_DECLARE_DEVICE_MATRIX_DATA(ValueType, IndexType) \
    struct device_matrix_data<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DEVICE_MATRIX_DATA);


}  // namespace gko

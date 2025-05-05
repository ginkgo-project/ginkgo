// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/device_matrix_data_kernels.hpp"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


// __half `!=` operation is only available in __device__
// Although gko::is_nonzero is constexpr, it still shows calling __device__ in
// __host__
template <typename T>
GKO_INLINE __device__ constexpr bool is_nonzero_(T value)
{
    return value != zero<T>();
}

template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  array<ValueType>& values, array<IndexType>& row_idxs,
                  array<IndexType>& col_idxs)
{
    using device_value_type = device_type<ValueType>;
    auto value_ptr = as_device_type(values.get_const_data());
    auto size = values.get_size();
    // count nonzeros
    auto nnz = thrust::count_if(
        thrust_policy(exec), value_ptr, value_ptr + size,
        [] __device__(device_value_type value) { return is_nonzero_(value); });
    if (nnz < size) {
        using tuple_type =
            thrust::tuple<IndexType, IndexType, device_value_type>;
        // allocate new storage
        array<ValueType> new_values{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_row_idxs{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_col_idxs{exec, static_cast<size_type>(nnz)};
        // copy nonzeros
        auto it = thrust::make_zip_iterator(thrust::make_tuple(
            row_idxs.get_const_data(), col_idxs.get_const_data(), value_ptr));
        auto out_it = thrust::make_zip_iterator(
            thrust::make_tuple(new_row_idxs.get_data(), new_col_idxs.get_data(),
                               as_device_type(new_values.get_data())));
        thrust::copy_if(thrust_policy(exec), it, it + size, out_it,
                        [] __device__(tuple_type entry) {
                            return is_nonzero_(thrust::get<2>(entry));
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
    const auto size = values.get_size();
    // CUDA 12.4 has a bug that requires these pointers to be non-const
    const auto rows = row_idxs.get_data();
    const auto cols = col_idxs.get_data();
    auto iota = thrust::make_counting_iterator(size_type{});
    const auto new_size = static_cast<size_type>(thrust::count_if(
        thrust_policy(exec), iota, iota + size,
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
        auto in_locs =
            thrust::make_zip_iterator(thrust::make_tuple(rows, cols));
        auto in_vals = as_device_type(values.get_const_data());
        auto out_locs = thrust::make_zip_iterator(thrust::make_tuple(
            new_row_idxs.get_data(), new_col_idxs.get_data()));
        auto out_vals = as_device_type(new_values.get_data());
        thrust::reduce_by_key(thrust_policy(exec), in_locs, in_locs + size,
                              in_vals, out_locs, out_vals);
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
                    size_type num_elems, IndexType* row_idxs,
                    IndexType* col_idxs, ValueType* vals)
{
    auto it = thrust::make_zip_iterator(thrust::make_tuple(row_idxs, col_idxs));
    auto vals_it = as_device_type(vals);
    thrust::sort_by_key(thrust_policy(exec), it, it + num_elems, vals_it);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

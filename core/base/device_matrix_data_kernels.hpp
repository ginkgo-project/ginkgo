// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_DEVICE_MATRIX_DATA_KERNELS_HPP_
#define GKO_CORE_BASE_DEVICE_MATRIX_DATA_KERNELS_HPP_


#include <ginkgo/core/base/device_matrix_data.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_DEVICE_MATRIX_DATA_SOA_TO_AOS_KERNEL(ValueType, IndexType) \
    void soa_to_aos(std::shared_ptr<const DefaultExecutor> exec,               \
                    const device_matrix_data<ValueType, IndexType>& in,        \
                    array<matrix_data_entry<ValueType, IndexType>>& out)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_AOS_TO_SOA_KERNEL(ValueType, IndexType) \
    void aos_to_soa(std::shared_ptr<const DefaultExecutor> exec,               \
                    const array<matrix_data_entry<ValueType, IndexType>>& in,  \
                    device_matrix_data<ValueType, IndexType>& out)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL(ValueType,       \
                                                           IndexType)       \
    void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,          \
                      array<ValueType>& values, array<IndexType>& row_idxs, \
                      array<IndexType>& col_idxs)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_SUM_DUPLICATES_KERNEL(ValueType, \
                                                             IndexType) \
    void sum_duplicates(std::shared_ptr<const DefaultExecutor> exec,    \
                        size_type num_rows, array<ValueType>& values,   \
                        array<IndexType>& row_idxs,                     \
                        array<IndexType>& col_idxs)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL(ValueType, \
                                                             IndexType) \
    void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,    \
                        device_matrix_data<ValueType, IndexType>& data)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                          \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DEVICE_MATRIX_DATA_SOA_TO_AOS_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DEVICE_MATRIX_DATA_AOS_TO_SOA_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DEVICE_MATRIX_DATA_SUM_DUPLICATES_KERNEL(ValueType,           \
                                                         IndexType);          \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_BASE_DEVICE_MATRIX_DATA_KERNELS_HPP_

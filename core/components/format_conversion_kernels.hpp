// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_FORMAT_CONVERSION_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_FORMAT_CONVERSION_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CONVERT_PTRS_TO_IDXS(IndexType, RowPtrType)             \
    void convert_ptrs_to_idxs(std::shared_ptr<const DefaultExecutor> exec,  \
                              const RowPtrType* ptrs, size_type num_blocks, \
                              IndexType* idxs)
#define GKO_DECLARE_CONVERT_PTRS_TO_IDXS32(IndexType) \
    GKO_DECLARE_CONVERT_PTRS_TO_IDXS(IndexType, ::gko::int32)
#define GKO_DECLARE_CONVERT_PTRS_TO_IDXS64(IndexType) \
    GKO_DECLARE_CONVERT_PTRS_TO_IDXS(IndexType, ::gko::int64)

#define GKO_DECLARE_CONVERT_IDXS_TO_PTRS(IndexType, RowPtrType)            \
    void convert_idxs_to_ptrs(std::shared_ptr<const DefaultExecutor> exec, \
                              const IndexType* idxs, size_type num_idxs,   \
                              size_type num_blocks, RowPtrType* ptrs)
#define GKO_DECLARE_CONVERT_IDXS_TO_PTRS32(IndexType) \
    GKO_DECLARE_CONVERT_IDXS_TO_PTRS(IndexType, ::gko::int32)
#define GKO_DECLARE_CONVERT_IDXS_TO_PTRS64(IndexType) \
    GKO_DECLARE_CONVERT_IDXS_TO_PTRS(IndexType, ::gko::int64)

#define GKO_DECLARE_CONVERT_PTRS_TO_SIZES(RowPtrType)                        \
    void convert_ptrs_to_sizes(std::shared_ptr<const DefaultExecutor> exec,  \
                               const RowPtrType* ptrs, size_type num_blocks, \
                               size_type* sizes)


#define GKO_DECLARE_ALL_AS_TEMPLATES                         \
    template <typename IndexType, typename RowPtrType>       \
    GKO_DECLARE_CONVERT_PTRS_TO_IDXS(IndexType, RowPtrType); \
    template <typename IndexType, typename RowPtrType>       \
    GKO_DECLARE_CONVERT_IDXS_TO_PTRS(IndexType, RowPtrType); \
    template <typename RowPtrType>                           \
    GKO_DECLARE_CONVERT_PTRS_TO_SIZES(RowPtrType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_FORMAT_CONVERSION_KERNELS_HPP_

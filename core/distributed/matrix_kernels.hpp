// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_MATRIX_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_MATRIX_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BUILD_LOCAL_NONLOCAL(ValueType, LocalIndexType,            \
                                         GlobalIndexType)                      \
    void build_local_nonlocal(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const device_matrix_data<ValueType, GlobalIndexType>& input,           \
        const experimental::distributed::Partition<                            \
            LocalIndexType, GlobalIndexType>* row_partition,                   \
        const experimental::distributed::Partition<                            \
            LocalIndexType, GlobalIndexType>* col_partition,                   \
        comm_index_type local_part, array<LocalIndexType>& local_row_idxs,     \
        array<LocalIndexType>& local_col_idxs, array<ValueType>& local_values, \
        array<LocalIndexType>& non_local_row_idxs,                             \
        array<LocalIndexType>& non_local_col_idxs,                             \
        array<ValueType>& non_local_values,                                    \
        array<LocalIndexType>& local_gather_idxs,                              \
        array<comm_index_type>& recv_offsets,                                  \
        array<GlobalIndexType>& non_local_to_global)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                    \
    using comm_index_type = experimental::distributed::comm_index_type; \
    template <typename ValueType, typename LocalIndexType,              \
              typename GlobalIndexType>                                 \
    GKO_DECLARE_BUILD_LOCAL_NONLOCAL(ValueType, LocalIndexType,         \
                                     GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(distributed_matrix,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_MATRIX_KERNELS_HPP_

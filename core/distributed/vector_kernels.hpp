// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_VECTOR_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_VECTOR_KERNELS_HPP_


// can't include ginkgo/core/distributed/vector.hpp since that requires linking
// against MPI
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_DISTRIBUTED_VECTOR_BUILD_LOCAL(ValueType, LocalIndexType, \
                                                   GlobalIndexType)           \
    void build_local(                                                         \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const device_matrix_data<ValueType, GlobalIndexType>& input,          \
        const experimental::distributed::Partition<                           \
            LocalIndexType, GlobalIndexType>* partition,                      \
        comm_index_type local_part, matrix::Dense<ValueType>* local_mtx)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                      \
    using comm_index_type = experimental::distributed::comm_index_type;   \
    template <typename ValueType, typename LocalIndexType,                \
              typename GlobalIndexType>                                   \
    GKO_DECLARE_DISTRIBUTED_VECTOR_BUILD_LOCAL(ValueType, LocalIndexType, \
                                               GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(distributed_vector,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_VECTOR_KERNELS_HPP_

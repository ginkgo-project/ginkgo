/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_CORE_DISTRIBUTED_PARTITION_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_PARTITION_KERNELS_HPP_


#include <ginkgo/core/distributed/partition.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_PARTITION_COUNT_RANGES                                 \
    void count_ranges(std::shared_ptr<const DefaultExecutor> exec, \
                      const Array<comm_index_type>& mapping,       \
                      size_type& num_ranges)

#define GKO_PARTITION_BUILD_FROM_CONTIGUOUS                                 \
    void build_from_contiguous(std::shared_ptr<const DefaultExecutor> exec, \
                               const Array<global_index_type>& ranges,      \
                               global_index_type* range_bounds,             \
                               comm_index_type* part_ids)

#define GKO_PARTITION_BUILD_FROM_MAPPING                                 \
    void build_from_mapping(std::shared_ptr<const DefaultExecutor> exec, \
                            const Array<comm_index_type>& mapping,       \
                            global_index_type* range_bounds,             \
                            comm_index_type* part_ids)

#define GKO_PARTITION_BUILD_FROM_GLOBAL_SIZE                      \
    void build_ranges_from_global_size(                           \
        std::shared_ptr<const DefaultExecutor> exec,              \
        comm_index_type num_parts, global_index_type global_size, \
        Array<global_index_type>& ranges)

#define GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES(LocalIndexType)          \
    void build_starting_indices(std::shared_ptr<const DefaultExecutor> exec,  \
                                const global_index_type* range_offsets,       \
                                const int* range_parts, size_type num_ranges, \
                                comm_index_type num_parts,                    \
                                comm_index_type& num_empty_parts,             \
                                LocalIndexType* ranks, LocalIndexType* sizes)

#define GKO_DECLARE_PARTITION_IS_ORDERED(LocalIndexType)                     \
    void is_ordered(std::shared_ptr<const DefaultExecutor> exec,             \
                    const distributed::Partition<LocalIndexType>* partition, \
                    bool* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                              \
    using global_index_type = distributed::global_index_type;     \
    using comm_index_type = distributed::comm_index_type;         \
    GKO_PARTITION_COUNT_RANGES;                                   \
    GKO_PARTITION_BUILD_FROM_CONTIGUOUS;                          \
    GKO_PARTITION_BUILD_FROM_MAPPING;                             \
    GKO_PARTITION_BUILD_FROM_GLOBAL_SIZE;                         \
    template <typename LocalIndexType>                            \
    GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES(LocalIndexType); \
    template <typename LocalIndexType>                            \
    GKO_DECLARE_PARTITION_IS_ORDERED(LocalIndexType)

GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(partition,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_PARTITION_KERNELS_HPP_

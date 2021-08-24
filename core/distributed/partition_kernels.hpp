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


namespace gko {
namespace kernels {


#define GKO_PARTITION_COUNT_RANGES                                 \
    void count_ranges(std::shared_ptr<const DefaultExecutor> exec, \
                      const Array<comm_index_type> &mapping,       \
                      size_type &num_ranges)

#define GKO_DECLARE_PARTITION_BUILD_FROM_CONTIGUOUS(LocalIndexType) \
    void build_from_contiguous(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                \
        const Array<global_index_type> &ranges,                     \
        distributed::Partition<LocalIndexType> *partition)

#define GKO_DECLARE_PARTITION_BUILD_FROM_MAPPING(LocalIndexType)         \
    void build_from_mapping(std::shared_ptr<const DefaultExecutor> exec, \
                            const Array<comm_index_type> &mapping,       \
                            distributed::Partition<LocalIndexType> *partition)

#define GKO_DECLARE_PARTITION_BUILD_RANKS(LocalIndexType)          \
    void build_ranks(std::shared_ptr<const DefaultExecutor> exec,  \
                     const global_index_type *range_offsets,       \
                     const int *range_parts, size_type num_ranges, \
                     int num_parts, LocalIndexType *ranks,         \
                     LocalIndexType *sizes)

#define GKO_DECLARE_PARTITION_IS_ORDERED(LocalIndexType)                     \
    void is_ordered(std::shared_ptr<const DefaultExecutor> exec,             \
                    const distributed::Partition<LocalIndexType> *partition, \
                    bool *result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                             \
    using global_index_type = distributed::global_index_type;    \
    using comm_index_type = distributed::comm_index_type;        \
    GKO_PARTITION_COUNT_RANGES;                                  \
    template <typename LocalIndexType>                           \
    GKO_DECLARE_PARTITION_BUILD_FROM_CONTIGUOUS(LocalIndexType); \
    template <typename LocalIndexType>                           \
    GKO_DECLARE_PARTITION_BUILD_FROM_MAPPING(LocalIndexType);    \
    template <typename LocalIndexType>                           \
    GKO_DECLARE_PARTITION_BUILD_RANKS(LocalIndexType);           \
    template <typename LocalIndexType>                           \
    GKO_DECLARE_PARTITION_IS_ORDERED(LocalIndexType)

namespace omp {
namespace partition {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace partition
}  // namespace omp


namespace cuda {
namespace partition {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace partition
}  // namespace cuda


namespace reference {
namespace partition {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace partition
}  // namespace reference


namespace hip {
namespace partition {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace partition
}  // namespace hip


namespace dpcpp {
namespace partition {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace partition
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_PARTITION_KERNELS_HPP_

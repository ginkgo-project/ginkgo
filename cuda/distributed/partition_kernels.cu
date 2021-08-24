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

#include "core/distributed/partition_kernels.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace partition {


void count_ranges(std::shared_ptr<const DefaultExecutor> exec,
                  const Array<comm_index_type> &mapping,
                  size_type &num_ranges) GKO_NOT_IMPLEMENTED;


template <typename LocalIndexType>
void build_from_contiguous(std::shared_ptr<const DefaultExecutor> exec,
                           const Array<global_index_type> &ranges,
                           distributed::Partition<LocalIndexType> *partition)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_BUILD_FROM_CONTIGUOUS);


template <typename LocalIndexType>
void build_from_mapping(std::shared_ptr<const DefaultExecutor> exec,
                        const Array<comm_index_type> &mapping,
                        distributed::Partition<LocalIndexType> *partition)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PARTITION_BUILD_FROM_MAPPING);


template <typename LocalIndexType>
void build_ranks(std::shared_ptr<const DefaultExecutor> exec,
                 const global_index_type *range_offsets, const int *range_parts,
                 size_type num_ranges, int num_parts, LocalIndexType *ranks,
                 LocalIndexType *sizes) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PARTITION_BUILD_RANKS);


template <typename LocalIndexType>
void is_ordered(std::shared_ptr<const DefaultExecutor> exec,
                const distributed::Partition<LocalIndexType> *partition,
                bool *result) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PARTITION_IS_ORDERED);


}  // namespace partition
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

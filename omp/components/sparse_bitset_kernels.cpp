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

#include "core/components/sparse_bitset_kernels.hpp"


#include <bitset>
#include <numeric>


#include <ginkgo/core/base/types.hpp>


#include "core/components/sparse_bitset.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace sparse_bitset {


template <typename GlobalIndexType>
void sort(std::shared_ptr<const DefaultExecutor> exec, GlobalIndexType* indices,
          size_type size)
{
    std::sort(indices, indices + size);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_SPARSE_BITSET_SORT_KERNEL);


template <typename GlobalIndexType>
void build_bitmap(std::shared_ptr<const DefaultExecutor> exec,
                  const GlobalIndexType* indices, size_type size,
                  uint32* bitmap, size_type num_blocks)
{
    std::fill_n(bitmap, num_blocks, uint32{});
    for (size_type i = 0; i < size; i++) {
        bitmap[i / sparse_bitset_word_size] |=
            uint32{1} << static_cast<uint32>(i % sparse_bitset_word_size);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SPARSE_BITSET_BUILD_BITMAP_KERNEL);


template <typename LocalIndexType>
void build_bitmap_ranks(std::shared_ptr<const DefaultExecutor> exec,
                        const uint32* bitmap, size_type num_blocks,
                        LocalIndexType* ranks)
{
    LocalIndexType count{};
    for (size_type i = 0; i < num_blocks; i++) {
        ranks[i] = count;
        count += std::bitset<sparse_bitset_word_size>(bitmap[i]).count();
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SPARSE_BITSET_BUILD_BITMAP_RANKS_KERNEL);


template <typename LocalIndexType, typename GlobalIndexType>
void build_multilevel(std::shared_ptr<const DefaultExecutor> exec,
                      const GlobalIndexType* values, size_type size,
                      array<uint32>& bitmaps, array<LocalIndexType>& ranks,
                      int depth, GlobalIndexType* offsets) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SPARSE_BITSET_BUILD_MULTILEVEL_KERNEL);


}  // namespace sparse_bitset
}  // namespace omp
}  // namespace kernels
}  // namespace gko

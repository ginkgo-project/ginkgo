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

#include "core/components/sparse_bitset.hpp"


#include <ginkgo/core/base/math.hpp>


#include "core/components/sparse_bitset_kernels.hpp"
#include "ginkgo/core/base/executor.hpp"


namespace gko {
namespace {


GKO_REGISTER_OPERATION(sort, sparse_bitset::sort);
GKO_REGISTER_OPERATION(build_bitmap, sparse_bitset::build_bitmap);
GKO_REGISTER_OPERATION(build_bitmap_ranks, sparse_bitset::build_bitmap_ranks);
GKO_REGISTER_OPERATION(build_multilevel, sparse_bitset::build_multilevel);


}  // namespace


template <int depth, typename LocalIndexType, typename GlobalIndexType>
sparse_bitset<depth, LocalIndexType, GlobalIndexType>::sparse_bitset(
    std::shared_ptr<const Executor> exec)
    : offsets{}, bitmaps{exec}, ranks{exec}
{}


template <int depth, typename LocalIndexType, typename GlobalIndexType>
sparse_bitset<depth, LocalIndexType, GlobalIndexType>
sparse_bitset<depth, LocalIndexType, GlobalIndexType>::from_indices_unsorted(
    array<GlobalIndexType> data, GlobalIndexType universe_size)
{
    data.get_executor()->run(make_sort(data.get_data(), data.get_num_elems()));
    return from_indices_sorted(data, universe_size);
}


template <int depth, typename LocalIndexType, typename GlobalIndexType>
sparse_bitset<depth, LocalIndexType, GlobalIndexType>
sparse_bitset<depth, LocalIndexType, GlobalIndexType>::from_indices_sorted(
    const array<GlobalIndexType>& data, GlobalIndexType universe_size)
{
    const auto exec = data.get_executor();
    sparse_bitset result(exec);
    if (depth == 0) {
        result.bitmaps.set_executor(exec);
        result.ranks.set_executor(exec);
        const auto num_blocks = static_cast<size_type>(
            ceildiv(universe_size, sparse_bitset_word_size));
        result.bitmaps.resize_and_reset(num_blocks);
        result.ranks.resize_and_reset(num_blocks);
        exec->run(make_build_bitmap(data.get_const_data(), data.get_num_elems(),
                                    result.bitmaps.get_data(), num_blocks));
        exec->run(make_build_bitmap_ranks(result.bitmaps.get_const_data(),
                                          num_blocks, result.ranks.get_data()));
    } else {
        exec->run(make_build_multilevel(
            data.get_const_data(), data.get_num_elems(), result.bitmaps,
            result.ranks, depth, result.offsets.data()));
    }
    return result;
}

template <int depth, typename LocalIndexType, typename GlobalIndexType>
device_sparse_bitset<depth, LocalIndexType, GlobalIndexType>
sparse_bitset<depth, LocalIndexType, GlobalIndexType>::to_device() const
{
    device_sparse_bitset<depth, LocalIndexType, GlobalIndexType> result;
    std::copy(this->offsets.begin(), this->offsets.end(), result.offsets.data);
    result.bitmaps = this->bitmaps.get_const_data();
    result.ranks = this->ranks.get_const_data();
    return result;
}


#define GKO_DECLARE_SPARSE_BITSET0(_local, _global) \
    class sparse_bitset<0, _local, _global>
#define GKO_DECLARE_SPARSE_BITSET1(_local, _global) \
    class sparse_bitset<1, _local, _global>
#define GKO_DECLARE_SPARSE_BITSET2(_local, _global) \
    class sparse_bitset<2, _local, _global>
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_SPARSE_BITSET0);
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_SPARSE_BITSET1);
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_SPARSE_BITSET2);


}  // namespace gko
/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

template <typename IndexType>
struct overflowing_sum {
    constexpr static IndexType max = std::numeric_limits<IndexType>::max();
    constexpr static IndexType sentinel = -1;

    constexpr IndexType operator()(IndexType i, IndexType j) const
    {
        if (did_overflow(i) || did_overflow(j) || max - i < j) {
            return sentinel;
        }
        return i + j;
    }

    constexpr static bool did_overflow(IndexType i) { return i < 0; }
};


template <>
struct overflowing_sum<size_type> {
    constexpr static size_type max = std::numeric_limits<size_type>::max();
    constexpr static size_type sentinel = max;

    constexpr size_type operator()(size_type i, size_type j) const
    {
        if (did_overflow(i) || did_overflow(j) || max - i < j) {
            return sentinel;
        }
        return i + j;
    }

    constexpr static bool did_overflow(size_type i) { return i == sentinel; }
};


template <typename IndexType>
void prefix_sum(std::shared_ptr<const DefaultExecutor> exec, IndexType* counts,
                size_type num_entries)
{
    constexpr auto max = std::numeric_limits<IndexType>::max();
    thrust::exclusive_scan(thrust_policy(exec), counts, counts + num_entries,
                           counts, 0, overflowing_sum<IndexType>{});
    if (num_entries > 0 &&
        overflowing_sum<IndexType>::did_overflow(
            exec->copy_val_to_host(counts + num_entries - 1))) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(IndexType)));
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PREFIX_SUM_KERNEL);

// instantiate for size_type as well, as this is used in the Sellp format
template void prefix_sum<size_type>(std::shared_ptr<const DefaultExecutor>,
                                    size_type*, size_type);

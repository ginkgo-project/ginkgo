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

#ifndef GKO_PUBLIC_CORE_MATRIX_CSR_LOOKUP_HPP_
#define GKO_PUBLIC_CORE_MATRIX_CSR_LOOKUP_HPP_


#include <type_traits>


#include <ginkgo/core/base/intrinsics.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace matrix {


/**
 * Type describing which kind of lookup structure is used to find entries in a
 * single row of a Csr matrix.
 */
enum class sparsity_type : int {
    /**
     * The row is dense, i.e. it contains all entries in
     * `[min_col, min_col + storage_size)`.
     * This means that the relative output index is `col - min_col`.
     */
    full = 1,
    /**
     * The row is sufficiently dense that its sparsity pattern can be stored in
     * a dense bitmap consisting of `storage_size` blocks of size `block_size`,
     * with a total of `storage_size` blocks. Each block stores its sparsity
     * pattern and the number of columns before. This means that the relative
     * output index can be computed as
     * ```
     * auto block = (col - min_col) / block_size;
     * auto local_col = (col - min_col) % block_size;
     * auto prefix_mask = (block_type{1} << local_col) - 1;
     * auto output_idx = base[block] + popcount(bitmap[block] & prefix_mask);
     * ```
     */
    bitmap = 2,
    /**
     * The row is sparse, so it is best represented using a hashtable.
     * The hashtable has size `storage_size` and stores the relative output
     * index directly, i.e.
     * ```
     * auto hash_key = col - min_col;
     * auto hash_bucket = hash(hash_key);
     * while (col_idxs[hashtable[hash_bucket]] != col) {
     *     hash_bucket = (hash_bucket + 1) % storage_size; // linear probing
     * }
     * auto output_idx = hashtable[hash_bucket];
     * ```
     */
    hash = 4,
};


inline sparsity_type operator|(sparsity_type a, sparsity_type b)
{
    return static_cast<sparsity_type>(static_cast<int>(a) |
                                      static_cast<int>(b));
}


template <typename IndexType>
class csr_sparsity_lookup {
    const IndexType* row_ptrs;
    const IndexType* col_idxs;
    Array<int64> row_desc;
    Array<int32> storage;
};


template <typename IndexType>
struct device_sparsity_lookup {
    /** Number of bits in a block_type entry. */
    static constexpr int block_size = 32;

    const IndexType* col_idxs;
    const IndexType row_nnz;
    const int32* storage;
    const int64 desc;

    GKO_ATTRIBUTES GKO_INLINE IndexType operator[](IndexType col) const
    {
        switch (static_cast<sparsity_type>(desc & 0xF)) {
        case sparsity_type::full:
            return lookup_full(col);
        case sparsity_type::bitmap:
            return lookup_bitmap(col);
        case sparsity_type::hash:
            return lookup_hash(col);
        }
        assert(false);
    }

    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_full(IndexType col) const
    {
        const auto min_col = col_idxs[0];
        const auto out_idx = col - min_col;
        assert(out_idx < row_nnz);
        assert(col_idxs[out_idx] == col);
        return out_idx;
    }

    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_bitmap(IndexType col) const
    {
        const auto min_col = col_idxs[0];
        const auto num_blocks = static_cast<int32>(desc >> 32);
        const auto block_bases = storage;
        const auto block_bitmaps =
            reinterpret_cast<const uint32*>(block_bases + num_blocks);
        const auto rel_col = col - min_col;
        const auto block = rel_col / block_size;
        const auto col_in_block = rel_col % block_size;
        const auto prefix_mask = (uint32{1} << col_in_block) - 1;
        assert(block < num_blocks);
        assert(block_bitmaps[block] & (uint32{1} << col_in_block));
        const auto out_idx =
            block_bases[block] +
            gko::detail::popcount(block_bitmaps[block] & prefix_mask);
        assert(col_idxs[out_idx] == col);
        return out_idx;
    }

    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_hash(IndexType col) const
    {
        const auto hashmap_size = static_cast<uint32>(2 * row_nnz);
        const auto hash_param = static_cast<uint32>(desc >> 32);
        const auto hashmap = storage;
        auto hash = (static_cast<uint32>(col) * hash_param) % hashmap_size;
        assert(hashmap[hash] >= 0);
        assert(hashmap[hash] < row_nnz);
        // linear probing with sentinel to avoid infinite loop
        while (col_idxs[hashmap[hash]] != col) {
            hash++;
            if (hash >= hashmap_size) {
                hash = 0;
            }
            assert(hashmap[hash] >= 0);
            assert(hashmap[hash] < row_nnz);
        }
        const auto out_idx = hashmap[hash];
        assert(col_idxs[out_idx] == col);
        return out_idx;
    }
};


}  // namespace matrix
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_MATRIX_CSR_LOOKUP_HPP_

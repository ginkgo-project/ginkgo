// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_CSR_LOOKUP_HPP_
#define GKO_CORE_MATRIX_CSR_LOOKUP_HPP_


#include <type_traits>


#include <ginkgo/core/base/intrinsics.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace matrix {
namespace csr {


/**
 * Type describing which kind of lookup structure is used to find entries in a
 * single row of a Csr matrix. It is also uses as a mask, so each value has only
 * a single bit set.
 */
enum class sparsity_type {
    /**
     * The row is dense, i.e. it contains all entries in
     * `[min_col, min_col + storage_size)`.
     * This means that the relative output index is `col - min_col`.
     */
    full = 1,
    /**
     * The row is sufficiently dense that its sparsity pattern can be stored in
     * a dense bitmap consisting of `storage_size` blocks of size `block_size`.
     * Each block stores its sparsity pattern as a bitmask and the number of
     * columns before the block as integer. This means that the relative output
     * index can be computed as
     * ```
     * auto block = (col - min_col) / sparsity_bitmap_block_size;
     * auto local_col = (col - min_col) % sparsity_bitmap_block_size;
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
     * auto hash_key = col;
     * auto hash_bucket = hash(hash_key);
     * while (local_cols[hashtable[hash_bucket]] != col) {
     *     hash_bucket = (hash_bucket + 1) % storage_size; // linear probing
     * }
     * auto output_idx = hashtable[hash_bucket];
     * ```
     */
    hash = 4,
};


GKO_ATTRIBUTES GKO_INLINE sparsity_type operator|(sparsity_type a,
                                                  sparsity_type b)
{
    return static_cast<sparsity_type>(static_cast<int>(a) |
                                      static_cast<int>(b));
}


GKO_ATTRIBUTES GKO_INLINE bool csr_lookup_allowed(sparsity_type allowed,
                                                  sparsity_type type)
{
    return ((static_cast<int>(allowed) & static_cast<int>(type)) != 0);
}


/** Number of bits in a block_type entry. */
static constexpr int sparsity_bitmap_block_size = 32;


template <typename IndexType>
struct device_sparsity_lookup {
    /**
     * Set up a device_sparsity_lookup structure from local data
     *
     * @param local_cols  the column array slice for the local row, it contains
     *                    row_nnz entries
     * @param row_nnz  the number of entries in this row
     * @param local_storage  the local lookup structure storage array slice, it
     *                       needs to be set up using the storage_offsets array
     * @param storage_size  the number of int32 entries in the local lookup
     *                      structure
     * @param desc  the lookup structure descriptor for this row
     */
    GKO_ATTRIBUTES GKO_INLINE device_sparsity_lookup(
        const IndexType* local_cols, IndexType row_nnz,
        const int32* local_storage, IndexType storage_size, int64 desc)
        : local_cols{local_cols},
          row_nnz{row_nnz},
          local_storage{local_storage},
          storage_size{storage_size},
          desc{desc}
    {}

    /**
     * Set up a device_sparsity_lookup structure from global data
     *
     * @param row_ptrs  the CSR row pointers
     * @param col_idxs  the CSR column indices
     * @param storage_offsets  the storage offset array for the lookup structure
     * @param storage  the storage array for the lookup data structure
     * @param descs  the lookup structure descriptor array
     * @param row  the row index to build the lookup structure for
     */
    GKO_ATTRIBUTES GKO_INLINE device_sparsity_lookup(
        const IndexType* row_ptrs, const IndexType* col_idxs,
        const IndexType* storage_offsets, const int32* storage,
        const int64* descs, size_type row)
    {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        const auto storage_begin = storage_offsets[row];
        const auto storage_end = storage_offsets[row + 1];
        local_cols = col_idxs + row_begin;
        row_nnz = row_end - row_begin;
        local_storage = storage + storage_begin;
        storage_size = storage_end - storage_begin;
        desc = descs[row];
    }

    /**
     * Returns the row-local index of the entry with the given column index, if
     * it is present.
     *
     * @param col  the column index
     * @return the row-local index of the given entry, or
     *         invalid_index<IndexType>() if it is not present.
     */
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
        GKO_ASSERT(false);
        return invalid_index<IndexType>();
    }

    /**
     * Returns the row-local index of the entry with the given column index,
     * assuming it is present.
     *
     * @param col  the column index
     * @return the row-local index of the given entry.
     * @note the function fails with an assertion (debug builds) or can crash if
     *       the column index is not present in the row!
     */
    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_unsafe(IndexType col) const
    {
        IndexType result{};
        switch (static_cast<sparsity_type>(desc & 0xF)) {
        case sparsity_type::full:
            result = lookup_full_unsafe(col);
            break;
        case sparsity_type::bitmap:
            result = lookup_bitmap_unsafe(col);
            break;
        case sparsity_type::hash:
            result = lookup_hash_unsafe(col);
            break;
        default:
            GKO_ASSERT(false);
        }
        GKO_ASSERT(local_cols[result] == col);
        return result;
    }

private:
    using unsigned_index_type = typename std::make_unsigned<IndexType>::type;

    const IndexType* local_cols;
    IndexType row_nnz;
    const int32* local_storage;
    IndexType storage_size;
    int64 desc;

    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_full_unsafe(IndexType col) const
    {
        const auto min_col = local_cols[0];
        const auto out_idx = col - min_col;
        GKO_ASSERT(out_idx >= 0 && out_idx < row_nnz);
        return out_idx;
    }

    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_full(IndexType col) const
    {
        const auto min_col = local_cols[0];
        const auto out_idx = col - min_col;
        return out_idx >= 0 && out_idx < row_nnz ? out_idx
                                                 : invalid_index<IndexType>();
    }

    GKO_ATTRIBUTES GKO_INLINE IndexType
    lookup_bitmap_unsafe(IndexType col) const
    {
        const auto min_col = local_cols[0];
        const auto num_blocks = static_cast<int32>(desc >> 32);
        const auto block_bases = local_storage;
        const auto block_bitmaps =
            reinterpret_cast<const uint32*>(block_bases + num_blocks);
        const auto rel_col = col - min_col;
        const auto block = rel_col / sparsity_bitmap_block_size;
        const auto col_in_block = rel_col % sparsity_bitmap_block_size;
        const auto prefix_mask = (uint32{1} << col_in_block) - 1;
        GKO_ASSERT(rel_col >= 0);
        GKO_ASSERT(block < num_blocks);
        GKO_ASSERT(block_bitmaps[block] & (uint32{1} << col_in_block));
        const auto out_idx =
            block_bases[block] +
            gko::detail::popcount(block_bitmaps[block] & prefix_mask);
        GKO_ASSERT(local_cols[out_idx] == col);
        return out_idx;
    }

    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_bitmap(IndexType col) const
    {
        const auto min_col = local_cols[0];
        const auto num_blocks = static_cast<int32>(desc >> 32);
        const auto block_bases = local_storage;
        const auto block_bitmaps =
            reinterpret_cast<const uint32*>(block_bases + num_blocks);
        const auto rel_col = col - min_col;
        const auto block = rel_col / sparsity_bitmap_block_size;
        const auto col_in_block = rel_col % sparsity_bitmap_block_size;
        if (rel_col < 0 || block >= num_blocks ||
            !(block_bitmaps[block] & (uint32{1} << col_in_block))) {
            return invalid_index<IndexType>();
        }
        const auto prefix_mask = (uint32{1} << col_in_block) - 1;
        return block_bases[block] +
               gko::detail::popcount(block_bitmaps[block] & prefix_mask);
    }

    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_hash_unsafe(IndexType col) const
    {
        const auto hashmap_size = static_cast<uint32>(storage_size);
        const auto hash_param = static_cast<uint32>(desc >> 32);
        const auto hashmap = local_storage;
        auto hash =
            (static_cast<unsigned_index_type>(col) * hash_param) % hashmap_size;
        GKO_ASSERT(hashmap[hash] >= 0);
        GKO_ASSERT(hashmap[hash] < row_nnz);
        while (local_cols[hashmap[hash]] != col) {
            hash++;
            if (hash >= hashmap_size) {
                hash = 0;
            }
            GKO_ASSERT(hashmap[hash] >= 0);
            GKO_ASSERT(hashmap[hash] < row_nnz);
        }
        const auto out_idx = hashmap[hash];
        return out_idx;
    }

    GKO_ATTRIBUTES GKO_INLINE IndexType lookup_hash(IndexType col) const
    {
        const auto hashmap_size = static_cast<uint32>(storage_size);
        const auto hash_param = static_cast<uint32>(desc >> 32);
        const auto hashmap = local_storage;
        auto hash =
            (static_cast<unsigned_index_type>(col) * hash_param) % hashmap_size;
        GKO_ASSERT(hashmap[hash] < row_nnz);
        auto out_idx = hashmap[hash];
        // linear probing with invalid_index sentinel to avoid infinite loop
        while (out_idx >= 0 && local_cols[out_idx] != col) {
            hash++;
            if (hash >= hashmap_size) {
                hash = 0;
            }
            out_idx = hashmap[hash];
            GKO_ASSERT(hashmap[hash] < row_nnz);
        }
        // out_idx is either correct or invalid_index, the hashmap sentinel
        return out_idx;
    }
};


}  // namespace csr
}  // namespace matrix
}  // namespace gko

#endif  // GKO_CORE_MATRIX_CSR_LOOKUP_HPP_

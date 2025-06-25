// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query.hpp"

#include <ginkgo/core/base/math.hpp>

#include "core/components/range_minimum_query_kernels.hpp"


namespace gko {
namespace {


GKO_REGISTER_OPERATION(compute_lookup_inside_blocks,
                       range_minimum_query::compute_lookup_inside_blocks);
GKO_REGISTER_OPERATION(compute_lookup_across_blocks,
                       range_minimum_query::compute_lookup_across_blocks);


}  // namespace


template <typename IndexType>
range_minimum_query<IndexType>::range_minimum_query(array<IndexType> data)
    : num_blocks_{static_cast<index_type>(
          ceildiv(static_cast<index_type>(data.get_size()), block_size))},
      lut_{data.get_executor()},
      block_tree_indices_{data.get_executor(),
                          static_cast<size_type>(num_blocks_)},
      block_argmin_storage_{
          data.get_executor(),
          static_cast<size_type>(block_argmin_view_type::storage_size(
              static_cast<size_type>(num_blocks_), block_argmin_num_bits))},
      block_min_{data.get_executor(), static_cast<size_type>(num_blocks_)},
      superblock_storage_{data.get_executor(),
                          static_cast<size_type>(
                              superblock_view_type::storage_size(num_blocks_))},
      values_{std::move(data)}
{
    const auto exec = values_.get_executor();
    auto block_argmin = block_argmin_view_type{
        block_argmin_storage_.get_data(), block_argmin_num_bits, num_blocks_};
    exec->run(make_compute_lookup_inside_blocks(
        values_.get_const_data(), static_cast<index_type>(values_.get_size()),
        block_argmin, block_min_.get_data(), block_tree_indices_.get_data()));
    auto superblocks =
        superblock_view_type{block_min_.get_const_data(),
                             superblock_storage_.get_data(), num_blocks_};
    exec->run(make_compute_lookup_across_blocks(block_min_.get_const_data(),
                                                num_blocks_, superblocks));
}


template <typename IndexType>
typename range_minimum_query<IndexType>::view_type
range_minimum_query<IndexType>::get() const
{
    return device_range_minimum_query{
        values_.get_const_data(),
        block_min_.get_const_data(),
        block_argmin_storage_.get_const_data(),
        block_tree_indices_.get_const_data(),
        superblock_storage_.get_const_data(),
        lut_.get(),
        static_cast<index_type>(values_.get_size())};
}


#define GKO_DEFINE_RANGE_MINIMUM_QUERY(IndexType) \
    class range_minimum_query<IndexType>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DEFINE_RANGE_MINIMUM_QUERY);


}  // namespace gko

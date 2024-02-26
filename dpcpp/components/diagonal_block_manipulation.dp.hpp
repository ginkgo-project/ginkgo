// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_DP_HPP_
#define GKO_DPCPP_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_DP_HPP_


#include <type_traits>


#include <CL/sycl.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace csr {


/**
 * @internal
 *
 * @note assumes that block dimensions are in "standard format":
 *       (subwarp_size, config::warp_size / subwarp_size, z)
 */
template <
    int max_block_size, int warps_per_block, typename Group, typename ValueType,
    typename IndexType,
    typename = std::enable_if_t<group::is_synchronizable_group<Group>::value>>
__dpct_inline__ void extract_transposed_diag_blocks(
    const Group& group, int processed_blocks,
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    ValueType* __restrict__ block_row, int increment,
    ValueType* __restrict__ workspace, sycl::nd_item<3> item_ct1)
{
    const int tid =
        item_ct1.get_local_id(1) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
    const auto warp = group::tiled_partition<config::warp_size>(group);
    auto bid = static_cast<size_type>(item_ct1.get_group(2)) * warps_per_block *
                   processed_blocks +
               item_ct1.get_local_id(0) * processed_blocks;
    auto bstart = (bid < num_blocks) ? block_ptrs[bid] : zero<IndexType>();
    IndexType bsize = 0;
#pragma unroll
    for (int b = 0; b < processed_blocks; ++b, ++bid) {
        if (bid < num_blocks) {
            bstart += bsize;
            bsize = block_ptrs[bid + 1] - bstart;
#pragma unroll
            for (int i = 0; i < max_block_size; ++i) {
                if (i < bsize) {
                    if (item_ct1.get_local_id(1) == b &&
                        item_ct1.get_local_id(2) < max_block_size) {
                        workspace[item_ct1.get_local_id(2)] = zero<ValueType>();
                    }
                    warp.sync();
                    const auto row = bstart + i;
                    const auto rstart = row_ptrs[row] + tid;
                    const auto rend = row_ptrs[row + 1];
                    // use the entire warp to ensure coalesced memory access
                    for (auto j = rstart; j < rend; j += config::warp_size) {
                        const auto col = col_idxs[j] - bstart;
                        if (col >= bsize) {
                            break;
                        }
                        if (col >= 0) {
                            workspace[col] = values[j];
                        }
                    }
                    warp.sync();
                    if (item_ct1.get_local_id(1) == b &&
                        item_ct1.get_local_id(2) < bsize) {
                        block_row[i * increment] =
                            workspace[item_ct1.get_local_id(2)];
                    }
                    warp.sync();
                }
            }
        }
    }
}


}  // namespace csr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_DP_HPP_

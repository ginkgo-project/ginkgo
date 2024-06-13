// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_CONFIG_HPP_
#define GKO_DPCPP_BASE_CONFIG_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


#include "core/base/types.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


struct config {
    /**
     * The type containing a bitmask over all lanes of a warp.
     */
    using lane_mask_type = uint64;

    /**
     * The number of threads within a Dpcpp subgroup.
     */
    static constexpr uint32 warp_size = 32;

    /**
     * The bitmask of the entire warp.
     */
    static constexpr auto full_lane_mask = ~zero<lane_mask_type>();

    /**
     * The minimal amount of warps that need to be scheduled for each block
     * to maximize GPU occupancy.
     */
    static constexpr uint32 min_warps_per_block = 4;

    /**
     * The default maximal number of threads allowed in DPCPP group
     */
    static constexpr uint32 max_block_size = 256;
};


/**
 * DCFG_1D provides the usual way to embed information from workgroup size and
 * sub_group size. We consider the workgroup size up to 4096 which requires 13
 * bits, and sub_group size up to 64 which requires 7 bits.
 */
using DCFG_1D = ConfigSet<13, 7>;


template <uint32 block, uint32 subgroup>
struct device_config {
    static constexpr uint32 block_size = block;
    static constexpr uint32 subgroup_size = subgroup;
    static constexpr uint32 encode = DCFG_1D::encode(block_size, subgroup_size);
};


/**
 * encode_list base type
 *
 * @tparam T  the input template
 */
template <typename T>
struct encode_list {};

/**
 * encode_list specializes for the type_list. It will convert the each type to
 * encoded information.
 *
 * @tparam T  the input template
 */
template <typename... Types>
struct encode_list<syn::type_list<Types...>> {
    using type = syn::value_list<uint32, Types::encode...>;
};


// dcfg_block_type_list_t is the type list for different workgroup size.
using dcfg_block_type_list_t =
    syn::type_list<device_config<512, 16>, device_config<256, 16>,
                   device_config<128, 16>>;

// dcfg_block_list_t is the value list variant of dcfg_block_type_list_t
using dcfg_block_list_t = encode_list<dcfg_block_type_list_t>::type;


// dcfg_1d_type_list_t is the type list for different workgroup and sub_group
// size.
using dcfg_1d_type_list_t =
    syn::type_list<device_config<512, 64>, device_config<512, 32>,
                   device_config<512, 16>, device_config<256, 32>,
                   device_config<256, 16>, device_config<256, 8>>;

// dcfg_1d_type_list_t is the value list variant of dcfg_1d_type_type_list_t
using dcfg_1d_list_t = encode_list<dcfg_1d_type_list_t>::type;


// dcfg_sq_type_list_t is the type list for different sub_group size and its
// workgroup size is square of sub_group.
using dcfg_sq_type_list_t =
    syn::type_list<device_config<4096, 64>, device_config<1024, 32>,
                   device_config<256, 16>, device_config<64, 8>>;

// dcfg_sq_list_t is the value list variant of dcfg_sq_type_list_t
using dcfg_sq_list_t = encode_list<dcfg_sq_type_list_t>::type;


// dcfg_1sg_list_t is the type list for only one sub_group in a workgroup.
using dcfg_1sg_type_list_t =
    syn::type_list<device_config<64, 64>, device_config<32, 32>,
                   device_config<16, 16>, device_config<8, 8>,
                   device_config<4, 4>>;

// dcfg_1sg_list_t is the value list variant of dcfg_1sg_type_list_t
using dcfg_1sg_list_t = encode_list<dcfg_1sg_type_list_t>::type;


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_CONFIG_HPP_

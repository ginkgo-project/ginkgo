// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_BLOCK_SIZES_HPP_
#define GKO_CORE_BASE_BLOCK_SIZES_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


namespace gko {
namespace fixedblock {


/**
 * @def GKO_FIXED_BLOCK_CUSTOM_SIZES
 * Optionally-defined comma-separated list of fixed block sizes to compile.
 */
#ifdef GKO_FIXED_BLOCK_CUSTOM_SIZES
/**
 * A compile-time list of block sizes for which dedicated fixed-block matrix
 * and corresponding preconditioner kernels should be compiled.
 */
using compiled_kernels = syn::value_list<int, GKO_FIXED_BLOCK_CUSTOM_SIZES>;
#else
using compiled_kernels = syn::value_list<int, 2, 3, 4, 7>;
#endif


}  // namespace fixedblock
}  // namespace gko


#endif  // GKO_CORE_BASE_BLOCK_SIZES_HPP_

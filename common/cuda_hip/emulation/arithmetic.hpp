// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_EMULATION_ARITHMETIC_HPP_
#define GKO_COMMON_CUDA_HIP_EMULATION_ARITHMETIC_HPP_


#include <ginkgo/config.hpp>

#include "core/emulation/double_word.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * The arithmetic of double in the accessor kernels with GINKGO_EMULATE_DOUBLE:
 * two float words with the accurate double-word algorithms and FMA, which GPUs
 * have. Special values are handled as for a single word, so that e.g. inf + 1
 * stays inf.
 */
using emulated_double =
    multi_word<float, 2,
               mw::policy<mw::normalization::eager, mw::algorithm::accurate,
                          true, mw::special::handle>>;


/**
 * The type that the accessor kernels compute in for values of type T. The
 * values stay T in memory, the accessor converts them when it reads and writes
 * them. With GINKGO_EMULATE_DOUBLE, double is computed in emulated_double,
 * every other type in itself.
 */
template <typename T>
struct accessor_arithmetic_impl {
    using type = T;
};

#if GINKGO_EMULATE_DOUBLE
template <>
struct accessor_arithmetic_impl<double> {
    using type = emulated_double;
};
#endif

template <typename T>
using accessor_arithmetic = typename accessor_arithmetic_impl<T>::type;


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_EMULATION_ARITHMETIC_HPP_

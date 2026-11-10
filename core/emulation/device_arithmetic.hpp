// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_EMULATION_DEVICE_ARITHMETIC_HPP_
#define GKO_CORE_EMULATION_DEVICE_ARITHMETIC_HPP_


#include <type_traits>

#include <ginkgo/config.hpp>

#include "core/emulation/double_word.hpp"


namespace gko {


/**
 * Double emulated by two float words, using the accurate double-word
 * algorithms with FMA and handling special values like a single word.
 */
using emulated_double =
    multi_word<float, 2,
               mw::policy<mw::normalization::eager, mw::algorithm::accurate,
                          true, mw::special::handle>>;


namespace detail {


template <typename T>
struct device_arithmetic {
    using type = T;
};

#if GINKGO_EMULATE_DOUBLE
template <>
struct device_arithmetic<double> {
    using type = emulated_double;
};
#endif


}  // namespace detail


/**
 * The type the CUDA and HIP kernels compute values of type `T` in. Unlike
 * device_type, it does not need to have the bit layout of `T`, since the
 * accessors convert the values when reading and writing them.
 *
 * @tparam T  the value type
 */
template <typename T>
using device_arithmetic_t =
    typename detail::device_arithmetic<std::remove_cv_t<T>>::type;


}  // namespace gko


#endif  // GKO_CORE_EMULATION_DEVICE_ARITHMETIC_HPP_

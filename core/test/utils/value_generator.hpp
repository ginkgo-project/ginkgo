// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_VALUE_GENERATOR_HPP_
#define GKO_CORE_TEST_UTILS_VALUE_GENERATOR_HPP_


#include <random>
#include <type_traits>


#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace test {
namespace detail {


/**
 * Generate a random value.
 *
 * @tparam ValueType  valuetype of the value
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param value_dist  distribution of array values
 * @param engine  a random engine
 *
 * @return ValueType
 */
template <typename ValueType, typename ValueDistribution, typename Engine>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(ValueDistribution&& value_dist, Engine&& gen)
{
    return value_dist(gen);
}

/**
 * Specialization for complex types.
 *
 * @copydoc get_rand_value
 */
template <typename ValueType, typename ValueDistribution, typename Engine>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(ValueDistribution&& value_dist, Engine&& gen)
{
    return ValueType(value_dist(gen), value_dist(gen));
}


}  // namespace detail
}  // namespace test
}  // namespace gko

#endif  // GKO_CORE_TEST_UTILS_VALUE_GENERATOR_HPP_

// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_ARRAY_GENERATOR_HPP_
#define GKO_CORE_TEST_UTILS_ARRAY_GENERATOR_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/test/utils/value_generator.hpp"


namespace gko {
namespace test {


/**
 * Generate a random array
 *
 * @tparam ValueType  valuetype of the array to generate
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param num  the number of elements of array
 * @param value_dist  distribution of array values
 * @param engine  a random engine
 * @param exec  executor where the array should be allocated
 *
 * @return array<ValueType>
 */
template <typename ValueType, typename ValueDistribution, typename Engine>
array<ValueType> generate_random_array(size_type num,
                                       ValueDistribution&& value_dist,
                                       Engine&& engine,
                                       std::shared_ptr<const Executor> exec)
{
    array<ValueType> array(exec->get_master(), num);
    auto val = array.get_data();
    for (int i = 0; i < num; i++) {
        val[i] = detail::get_rand_value<ValueType>(value_dist, engine);
    }
    array.set_executor(exec);
    return array;
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_ARRAY_GENERATOR_HPP_

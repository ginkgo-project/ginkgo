// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_TYPE_DESCRIPTOR_HELPER_HPP_
#define GKO_CORE_CONFIG_TYPE_DESCRIPTOR_HELPER_HPP_


#include <string>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace config {


/**
 This function updates the default type setting from current config. Any type
 that is not specified in the config will fall back to the type stored in the
 current type_descriptor.
 */
type_descriptor update_type(const pnode& config, const type_descriptor& td);


// type_string providing the mapping from type to string.
template <typename T>
struct type_string {};

#define TYPE_STRING_OVERLOAD(_type, _str)         \
    template <>                                   \
    struct type_string<_type> {                   \
        static std::string str() { return _str; } \
    }

TYPE_STRING_OVERLOAD(void, "void");
TYPE_STRING_OVERLOAD(double, "float64");
TYPE_STRING_OVERLOAD(float, "float32");
TYPE_STRING_OVERLOAD(std::complex<double>, "complex<float64>");
TYPE_STRING_OVERLOAD(std::complex<float>, "complex<float32>");
TYPE_STRING_OVERLOAD(int32, "int32");
TYPE_STRING_OVERLOAD(int64, "int64");


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_TYPE_DESCRIPTOR_HELPER_HPP_

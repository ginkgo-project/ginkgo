// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_VALIDATION_HPP_
#define GKO_CORE_BASE_VALIDATION_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>


namespace gko {
namespace validation {


#define GKO_VALIDATE(_expression, _message)                                 \
    if (!(_expression)) {                                                   \
        throw gko::InvalidData(__FILE__, __LINE__, typeid(decltype(*this)), \
                               _message);                                   \
    }


template <typename IndexType>
bool is_sorted(const gko::array<IndexType>& values)
{
    const auto host_values =
        make_temporary_clone(values.get_executor()->get_master(), &values);
    return std::is_sorted(
        host_values->get_const_data(),
        host_values->get_const_data() + host_values->get_size());
}


}  // namespace validation
}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_

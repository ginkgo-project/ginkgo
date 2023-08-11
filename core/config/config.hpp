/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_CONFIG_CONFIG_HPP_
#define GKO_CORE_CONFIG_CONFIG_HPP_


#include <string>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/stop/criterion.hpp>

namespace gko {
namespace config {

/**
 * This function is to update the default type setting from current config.
 *
 * @note It might update the unused type for the current class.
 */
inline TypeDescriptor update_type(const Config& config,
                                  const TypeDescriptor& td)
{
    TypeDescriptor updated = td;
    auto it = config.find("ValueType");
    if (it != config.end()) {
        updated.first = it->second;
    }
    it = config.find("IndexType");
    if (it != config.end()) {
        updated.second = it->second;
    }
    return updated;
}


using item = std::string;

template <typename T>
inline std::shared_ptr<T> get_pointer(const item& item, const registry& context,
                                      std::shared_ptr<const Executor> exec,
                                      TypeDescriptor td)
{
    std::shared_ptr<T> ptr;
    using T_non_const = std::remove_const_t<T>;
    ptr = context.search_data<T_non_const>(item);
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
inline std::shared_ptr<const LinOpFactory> get_pointer<const LinOpFactory>(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    std::shared_ptr<const LinOpFactory> ptr;
    ptr = context.search_data<LinOpFactory>(item);
    // handle object is item
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}


template <>
inline std::shared_ptr<const stop::CriterionFactory>
get_pointer<const stop::CriterionFactory>(const item& item,
                                          const registry& context,
                                          std::shared_ptr<const Executor> exec,
                                          TypeDescriptor td)
{
    std::shared_ptr<const stop::CriterionFactory> ptr;
    ptr = context.search_data<stop::CriterionFactory>(item);
    // handle object is item
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}


template <typename T>
inline std::vector<std::shared_ptr<T>> get_pointer_vector(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    std::vector<std::shared_ptr<T>> res;
    // for loop in item
    res.push_back(get_pointer<T>(item, context, exec, td));
    return res;
}

#define SET_POINTER(_factory, _param_type, _param_name, _config, _context,     \
                    _exec, _td)                                                \
    {                                                                          \
        auto it = _config.find(#_param_name);                                  \
        if (it != _config.end()) {                                             \
            _factory.with_##_param_name(gko::config::get_pointer<_param_type>( \
                it->second, _context, _exec, _td));                            \
        }                                                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_CONFIG_HPP_

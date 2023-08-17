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

#ifndef GKO_PUBLIC_CORE_CONFIG_REGISTRY_HPP_
#define GKO_PUBLIC_CORE_CONFIG_REGISTRY_HPP_


#include <complex>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace config {


class registry;


using linop_map = std::unordered_map<std::string, std::shared_ptr<LinOp>>;
using linopfactory_map =
    std::unordered_map<std::string, std::shared_ptr<LinOpFactory>>;
using criterionfactory_map =
    std::unordered_map<std::string, std::shared_ptr<stop::CriterionFactory>>;
using type_descriptor = std::pair<std::string, std::string>;
using buildfromconfig_map =
    std::map<std::string,
             std::function<std::unique_ptr<gko::LinOpFactory>(
                 const pnode&, const registry&,
                 std::shared_ptr<const Executor>&, type_descriptor)>>;


/**
 * map_type gives the map type according to the base type of given type.
 *
 * @tparam T  the type
 */
template <typename T, typename = void>
struct map_type {
    using type = void;
};

template <typename T>
struct map_type<
    T, typename std::enable_if<std::is_convertible<T*, LinOp*>::value>::type> {
    using type = linop_map;
};

template <typename T>
struct map_type<T, typename std::enable_if<
                       std::is_convertible<T*, LinOpFactory*>::value>::type> {
    using type = linopfactory_map;
};

template <typename T>
struct map_type<T, typename std::enable_if<std::is_convertible<
                       T*, stop::CriterionFactory*>::value>::type> {
    using type = criterionfactory_map;
};


class registry {
public:
    registry(buildfromconfig_map build_map) : build_map_(build_map) {}

    /**
     * insert_data stores the data with the key.
     *
     * @tparam T  the type
     *
     * @param key  the unique key string
     * @param data  the shared pointer of the object
     */
    template <typename T>
    bool emplace(std::string key, std::shared_ptr<T> data)
    {
        auto it = this->get_map<T>().emplace(key, data);
        return it.second;
    }

    /**
     * search_data searches the key on the corresponding map.
     *
     * @tparam T  the type
     *
     * @param key  the key string
     *
     * @return the shared pointer of the object
     */
    template <typename T>
    std::shared_ptr<T> search_data(std::string key) const
    {
        return gko::as<T>(this->get_map<T>().at(key));
    }

    const buildfromconfig_map& get_build_map() const { return build_map_; }

protected:
    /**
     * get_map gets the member map
     *
     * @tparam T  the type
     *
     * @return the map
     */
    template <typename T>
    typename map_type<T>::type& get_map()
    {
        return this->get_map_impl<typename map_type<T>::type>();
    }

    template <typename T>
    const typename map_type<T>::type& get_map() const
    {
        return this->get_map_impl<typename map_type<T>::type>();
    }

    /**
     * get_map_impl is the implementation of get_map
     *
     * @tparam T  the map type
     *
     * @return the map
     */
    template <typename T>
    T& get_map_impl();

    template <typename T>
    const T& get_map_impl() const;

private:
    linop_map linop_map_;
    linopfactory_map linopfactory_map_;
    criterionfactory_map criterionfactory_map_;
    buildfromconfig_map build_map_;
};


template <>
inline linop_map& registry::get_map_impl<linop_map>()
{
    return linop_map_;
}

template <>
inline linopfactory_map& registry::get_map_impl<linopfactory_map>()
{
    return linopfactory_map_;
}

template <>
inline criterionfactory_map& registry::get_map_impl<criterionfactory_map>()
{
    return criterionfactory_map_;
}

template <>
inline const linop_map& registry::get_map_impl<linop_map>() const
{
    return linop_map_;
}

template <>
inline const linopfactory_map& registry::get_map_impl<linopfactory_map>() const
{
    return linopfactory_map_;
}

template <>
inline const criterionfactory_map&
registry::get_map_impl<criterionfactory_map>() const
{
    return criterionfactory_map_;
}


}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_REGISTRY_HPP_

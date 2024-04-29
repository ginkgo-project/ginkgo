// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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

class type_descriptor;


using configuration_map =
    std::map<std::string,
             std::function<deferred_factory_parameter<gko::LinOpFactory>(
                 const pnode&, const registry&, type_descriptor)>>;


/**
 * This class stores additional context for creating Ginkgo objects from
 * configuration files.
 *
 * The context can contain user provided objects of the following types:
 * - LinOp
 * - LinOpFactory
 * - CriterionFactory
 *
 * Additionally, users can provide mappings from a configuration (provided as
 * a pnode) to user-defined types that are derived from one of the previous
 * stated types.
 */
class registry {
private:
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
    struct map_type<T, typename std::enable_if<
                           std::is_convertible<T*, LinOp*>::value>::type> {
        using type = std::unordered_map<std::string, std::shared_ptr<LinOp>>;
    };

    template <typename T>
    struct map_type<T, typename std::enable_if<std::is_convertible<
                           T*, LinOpFactory*>::value>::type> {
        using type =
            std::unordered_map<std::string, std::shared_ptr<LinOpFactory>>;
    };

    template <typename T>
    struct map_type<T, typename std::enable_if<std::is_convertible<
                           T*, stop::CriterionFactory*>::value>::type> {
        using type =
            std::unordered_map<std::string,
                               std::shared_ptr<stop::CriterionFactory>>;
    };

public:
    /**
     * registry constructor
     *
     * @param build_map  the build map to dispatch the class base. Ginkgo
     * provides `generate_config_map()` in config.hpp to provide the ginkgo
     * build map. Users can extend this map to fit their own LinOpFactory.
     */
    registry(configuration_map build_map) : build_map_(build_map) {}

    /**
     * insert_data stores the data with the key.
     *
     * @tparam T  the type
     *
     * @param key  the unique key string
     * @param data  the shared pointer of the object
     */
    template <typename T>
    bool emplace(std::string key, std::shared_ptr<T> data);

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
    std::shared_ptr<T> search_data(std::string key) const;

    /**
     * get the stored build map
     */
    const configuration_map& get_build_map() const { return build_map_; }

protected:
    /**
     * get_map gets the member map
     *
     * @tparam T  the type
     *
     * @return the map
     */
    template <typename T>
    typename map_type<T>::type& get_map();

    template <typename T>
    const typename map_type<T>::type& get_map() const;

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
    std::unordered_map<std::string, std::shared_ptr<LinOp>> linop_map_;
    std::unordered_map<std::string, std::shared_ptr<LinOpFactory>>
        linopfactory_map_;
    std::unordered_map<std::string, std::shared_ptr<stop::CriterionFactory>>
        criterionfactory_map_;
    configuration_map build_map_;
};


template <typename T>
inline bool registry::emplace(std::string key, std::shared_ptr<T> data)
{
    auto it = this->get_map<T>().emplace(key, data);
    return it.second;
}


template <typename T>
inline std::shared_ptr<T> registry::search_data(std::string key) const
{
    return gko::as<T>(this->get_map<T>().at(key));
}


template <typename T>
inline typename registry::map_type<T>::type& registry::get_map()
{
    return this->get_map_impl<typename map_type<T>::type>();
}

template <typename T>
inline const typename registry::map_type<T>::type& registry::get_map() const
{
    return this->get_map_impl<typename map_type<T>::type>();
}


}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_REGISTRY_HPP_

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

#include <ginkgo/core/base/exception_helpers.hpp>
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


namespace detail {


class registry_accessor;


/**
 * base_type gives the base type according to given type.
 *
 * @tparam T  the type
 */
template <typename T, typename = void>
struct base_type {};

template <typename T>
struct base_type<T, std::enable_if_t<std::is_convertible<T*, LinOp*>::value>> {
    using type = LinOp;
};

template <typename T>
struct base_type<
    T, std::enable_if_t<std::is_convertible<T*, LinOpFactory*>::value>> {
    using type = LinOpFactory;
};

template <typename T>
struct base_type<
    T,
    std::enable_if_t<std::is_convertible<T*, stop::CriterionFactory*>::value>> {
    using type = stop::CriterionFactory;
};


/**
 * allowed_ptr is a type-erased object for LinOp/LinOpFactory/CriterionFactory
 * shared_ptr.
 */
class allowed_ptr {
public:
    /**
     * The constructor accepts any shared pointer whose base type is LinOp,
     * LinOpFactory, or CriterionFactory. We use a template rather than
     * a constructor without a template because it allows the user to directly
     * use uninitialized_list in the registry constructor without wrapping
     * allowed_ptr manually.
     */
    template <typename Type>
    allowed_ptr(std::shared_ptr<Type> obj);

    /**
     * Check whether it contains the Type data
     *
     * @tparam Type  the checking type
     */
    template <typename Type>
    bool contains() const;

    /**
     * Get the shared pointer with Type
     *
     * @tparam Type  the desired type
     *
     * @return the shared pointer of Type
     */
    template <typename Type>
    std::shared_ptr<Type> get() const;

private:
    struct generic_container {
        virtual ~generic_container() = default;
    };

    template <typename Type>
    struct concrete_container : generic_container {
        concrete_container(std::shared_ptr<Type> obj) : ptr{obj}
        {
            static_assert(
                std::is_same<Type, typename base_type<Type>::type>::value,
                "The given type must be a base_type");
        }

        std::shared_ptr<Type> ptr;
    };

    std::shared_ptr<generic_container> data_;
};


template <typename Type>
inline allowed_ptr::allowed_ptr(std::shared_ptr<Type> obj)
{
    data_ =
        std::make_shared<concrete_container<typename base_type<Type>::type>>(
            obj);
}


template <typename Type>
inline bool allowed_ptr::contains() const
{
    return dynamic_cast<const concrete_container<Type>*>(data_.get());
}


template <typename Type>
inline std::shared_ptr<Type> allowed_ptr::get() const
{
    GKO_THROW_IF_INVALID(this->template contains<Type>(),
                         "does not hold the requested type.");
    return dynamic_cast<concrete_container<Type>*>(data_.get())->ptr;
}


}  // namespace detail


/**
 * This class stores additional context for creating Ginkgo objects from
 * configuration files.
 *
 * The context can contain user-provided objects that derive from the following
 * base types:
 * - LinOp
 * - LinOpFactory
 * - CriterionFactory
 *
 * Additionally, users can provide mappings from a configuration (provided as
 * a pnode) to user-defined types that are derived from LinOpFactory
 */
class registry final {
public:
    friend class detail::registry_accessor;


    /**
     * registry constructor
     *
     * @param additional_map  the additional map to dispatch the class base.
     *                        Users can extend the map to fit their own
     *                        LinOpFactory. We suggest using "usr::" as the
     *                        prefix in the key to simply avoid conflict with
     *                        ginkgo's map.
     */
    registry(const configuration_map& additional_map = {});

    /**
     * registry constructor
     *
     * @param stored_map  the map stores the shared pointer of users' objects.
     *                    It can hold any type whose base type is
     *                    LinOp/LinOpFactory/CriterionFactory.
     * @param additional_map  the additional map to dispatch the class base.
     *                        Users can extend the map to fit their own
     *                        LinOpFactory. We suggest using "usr::" as the
     *                        prefix in the key to simply avoid conflict with
     *                        ginkgo's map.
     */
    registry(
        const std::unordered_map<std::string, detail::allowed_ptr>& stored_map,
        const configuration_map& additional_map = {});

    /**
     * Store the data with the key.
     *
     * @tparam T  the type
     *
     * @param key  the unique key string
     * @param data  the shared pointer of the object
     */
    template <typename T>
    bool emplace(std::string key, std::shared_ptr<T> data);

protected:
    /**
     * Search the key on the corresponding map.
     *
     * @tparam T  the type
     *
     * @param key  the key string
     *
     * @return the shared pointer of the object
     */
    template <typename T>
    std::shared_ptr<T> get_data(std::string key) const;

    /**
     * Get the stored build map
     */
    const configuration_map& get_build_map() const { return build_map_; }

private:
    std::unordered_map<std::string, detail::allowed_ptr> stored_map_;
    configuration_map build_map_;
};


template <typename T>
inline bool registry::emplace(std::string key, std::shared_ptr<T> data)
{
    auto it = stored_map_.emplace(key, data);
    return it.second;
}


template <typename T>
inline std::shared_ptr<T> registry::get_data(std::string key) const
{
    return gko::as<T>(stored_map_.at(key)
                          .template get<typename detail::base_type<T>::type>());
}

}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_REGISTRY_HPP_

// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_TYPE_DESCRIPTOR_HPP_
#define GKO_PUBLIC_CORE_CONFIG_TYPE_DESCRIPTOR_HPP_


#include <string>

#include <ginkgo/core/base/types.hpp>

namespace gko {
namespace config {


/**
 * This class describes the value and index types to be used when building a
 * Ginkgo type from a configuration file.
 *
 * A type_descriptor is passed in order to define the parse function defines
 * which template parameters, in terms of value_type and/or index_type, the
 * created object will have. For example, a CG solver created like this:
 * ```
 * auto cg = parse(config, context, type_descriptor("float64", "int32"));
 * ```
 * will have the value type `float64` and the index type `int32`. Any Ginkgo
 * object that does not require one of these types will just ignore it. In
 * value_type, one additional value `void` can be used to specify that no
 * default type is provided. In this case, the configuration has to provide the
 * necessary template types.
 *
 * If the configuration specifies one field (only allow value_type now):
 * ```
 * value_type: "some_value_type"
 * ```
 * this type will take precedence over the type_descriptor.
 */
class type_descriptor final {
public:
    /**
     * type_descriptor constructor. There is free function
     * `make_type_descriptor` to create the object by template.
     *
     * @param value_typestr  the value type string. "void" means no default.
     * @param index_typestr  the (local) index type string.
     * @param global_index_typestr  the global index type string.
     *
     * @note there is no way to call the constructor with explicit template, so
     * we create another free function to handle it.
     */
    explicit type_descriptor(std::string value_typestr = "float64",
                             std::string index_typestr = "int32",
                             std::string global_index_typestr = "int64");

    /**
     * Get the value type string.
     */
    const std::string& get_value_typestr() const;

    /**
     * Get the index type string
     */
    const std::string& get_index_typestr() const;

    /**
     * Get the local index type string, which gives the same result as
     * get_index_typestr()
     */
    const std::string& get_local_index_typestr() const;

    /**
     * Get the global index type string
     */
    const std::string& get_global_index_typestr() const;

private:
    std::string value_typestr_;
    std::string index_typestr_;
    std::string global_index_typestr_;
};


/**
 * A helper function to properly set up the descriptor
 * from template type directly.
 *
 * @tparam ValueType  the value type in descriptor
 * @tparam IndexType  the index type in descriptor
 * @tparam GlobalIndexType  the global index type in descriptor
 */
template <typename ValueType = double, typename IndexType = int32,
          typename GlobalIndexType = int64>
type_descriptor make_type_descriptor();


}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_TYPE_DESCRIPTOR_HPP_

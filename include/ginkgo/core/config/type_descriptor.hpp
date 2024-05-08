// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_TYPE_DESCRIPTOR_HPP_
#define GKO_PUBLIC_CORE_CONFIG_TYPE_DESCRIPTOR_HPP_


#include <string>

namespace gko {
namespace config {


/**
 * This class describes the value and index types to be used when building a
 * Ginkgo type from a configuration file.
 *
 * A type_descriptor is passed to the parse function defines which
 * template parameters, in terms of value_type and/or index_type, the created
 * object will have. For example, a CG solver created like this:
 * ```
 * auto cg = parse(config, context, type_descriptor("float64", "int32"));
 * ```
 * will have the value type `float64` and the index type `int32`. Any Ginkgo
 * object that does not require one of these types will just ignore it. We used
 * void type to specify no default type.
 *
 * If the configurations specifies one of the fields (or both):
 * ```
 * value_type: "some_value_type"
 * index_type: "some_index_type"
 * ```
 * these types will take precedence over the type_descriptor.
 */
class type_descriptor final {
public:
    /**
     * type_descriptor constructor. There is free function
     * `make_type_descriptor` to create the object by template.
     *
     * @param value_typestr  the value type string. "void" means no default.
     * @param index_typestr  the index type string. "void" means no default.
     *
     * @note there is no way to call the constructor with explicit template, so
     * we create another free function to handle it.
     */
    explicit type_descriptor(std::string value_typestr = "float64",
                             std::string index_typestr = "int32");

    /**
     * Get the value type string.
     */
    const std::string& get_value_typestr() const;

    /**
     * Get the index type string
     */
    const std::string& get_index_typestr() const;

private:
    std::string value_typestr_;
    std::string index_typestr_;
};


/**
 * make_type_descriptor is a helper function to properly set up the descriptor
 * from template type directly.
 *
 * @tparam ValueType  the value type in descriptor
 * @tparam IndexType  the index type in descriptor
 */
template <typename ValueType = double, typename IndexType = int>
type_descriptor make_type_descriptor();


}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_TYPE_DESCRIPTOR_HPP_

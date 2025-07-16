// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_
#define GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_


#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


namespace gko {
namespace config {


/**
 * pnode describes a tree of properties.
 *
 * A pnode can either be empty, hold a value (a string, integer, real, or bool),
 * contain an array of pnode., or contain a mapping between strings and pnodes.
 */
class pnode final {
public:
    using key_type = std::string;
    using map_type = std::map<key_type, pnode>;
    using array_type = std::vector<pnode>;

    /**
     * tag_t is the indicator for the current node storage.
     */
    enum class tag_t { empty, array, boolean, real, integer, string, map };

    /**
     * Default constructor: create an empty node
     */
    explicit pnode();

    /**
     * Constructor for bool
     *
     * @param boolean  the bool type value
     */
    explicit pnode(bool boolean);

    /**
     * Constructor for integer with all integer type
     *
     * @tparam T  input type
     *
     * @param integer  the integer type value
     */
    template <typename T,
              std::enable_if_t<std::is_integral<T>::value>* = nullptr>
    explicit pnode(T integer);

    /**
     * Constructor for string
     *
     * @param str  string type value
     */
    explicit pnode(const std::string& str);

    /**
     * Constructor for char* (otherwise, it will use bool)
     *
     * @param str  the string like "..."
     */
    explicit pnode(const char* str);

    /**
     * Constructor for double (and also float)
     *
     * @param real  the floating point type value
     */
    explicit pnode(double real);

    /**
     * Constructor for array
     *
     * @param array  an pnode array
     */
    explicit pnode(const array_type& array);

    /**
     * Constructor for map
     *
     * @param map  a (string, pnode)-map
     */
    explicit pnode(const map_type& map);

    /**
     * bool conversion. It's true if and only if it is not empty.
     */
    explicit operator bool() const noexcept;

    /**
     * Check whether the representing data of two pnodes are the same
     */
    bool operator==(const pnode& rhs) const;

    /**
     * Check whether the representing data of two pnodes are different.
     */
    bool operator!=(const pnode& rhs) const;

    /**
     * Get the current node tag.
     *
     * @return the tag
     */
    tag_t get_tag() const;

    /**
     * Access the array stored in this property node. Throws
     * `gko::InvalidStateError` if the property node does not store an array.
     *
     * @return the array
     */
    const array_type& get_array() const;

    /**
     * Access the map stored in this property node. Throws
     * `gko::InvalidStateError` if the property node does not store a map.
     *
     * @return the map
     */
    const map_type& get_map() const;

    /**
     * Access the boolean value stored in this property node. Throws
     * `gko::InvalidStateError` if the property node does not store a boolean
     * value.
     *
     * @return the boolean value
     */
    bool get_boolean() const;

    /**
     * * Access the integer value stored in this property node. Throws
     * `gko::InvalidStateError` if the property node does not store an integer
     * value.
     *
     * @return the integer value
     */
    std::int64_t get_integer() const;

    /**
     * Access the real floating point value stored in this property node. Throws
     * `gko::InvalidStateError` if the property node does not store a real value
     *
     * @return the real floating point value
     */
    double get_real() const;

    /**
     * Access the string stored in this property node. Throws
     * `gko::InvalidStateError` if the property node does not store a string.
     *
     * @return the string
     */
    const std::string& get_string() const;

    /**
     * This function is to access the data under the map. It will throw error
     * when it does not hold a map. When access non-existent key in the map, it
     * will return an empty node.
     *
     * @param key  the key for the node of the map
     *
     * @return node. If the map does not have the key, return
     * an empty node.
     */
    const pnode& get(const std::string& key) const;

    /**
     * This function is to access the data under the array. It will throw error
     * when it does not hold an array or access out-of-bound index.
     *
     * @param index  the node index in array
     *
     * @return node.
     */
    const pnode& get(int index) const;

private:
    void throw_if_not_contain(tag_t tag) const;

    static const pnode& empty_node();

    tag_t tag_;
    array_type array_;  // for array
    map_type map_;      // for map
    // value
    std::string str_;
    union {
        std::int64_t integer_;
        double real_;
        bool boolean_;
    } union_data_;
};


template <typename T, std::enable_if_t<std::is_integral<T>::value>*>
pnode::pnode(T integer) : tag_(tag_t::integer)
{
    if (integer > std::numeric_limits<std::int64_t>::max() ||
        (std::is_signed<T>::value &&
         integer < std::numeric_limits<std::int64_t>::min())) {
        throw std::runtime_error("The input is out of the range of int64_t.");
    }
    union_data_.integer_ = static_cast<std::int64_t>(integer);
}


}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_

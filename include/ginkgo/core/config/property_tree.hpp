// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_
#define GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_


#include <cassert>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


namespace gko {
namespace config {


/**
 * pnode is to describe the property tree.
 */
class pnode {
public:
    using key_type = std::string;
    using map_type = std::map<key_type, pnode>;
    using array_type = std::vector<pnode>;

    /**
     * status_t is the indicator for the current node storage.
     */
    enum class status_t { empty, array, boolean, real, integer, string, map };

    /**
     * Default constructor: create an empty node
     */
    pnode();

    /**
     * Constructor for bool
     *
     * @param boolean  the bool type value
     */
    pnode(bool boolean);

    /**
     * Constructor for integer with all integer type
     *
     * @tparam T  input type
     *
     * @param integer  the integer type value
     */
    template <typename T, typename = typename std::enable_if<
                              std::is_integral<T>::value>::type>
    pnode(T integer);

    /**
     * Constructor for string
     *
     * @param str  string type value
     */
    pnode(const std::string& str);

    /**
     * Constructor for char* (otherwise, it will use bool)
     *
     * @param str  the string like "..."
     */
    pnode(const char* str);

    /**
     * Constructor for double
     *
     * @param real  the floating point type value
     */
    pnode(double real);

    /**
     * Constructor for array
     *
     * @param array  an pnode array
     */
    pnode(const array_type& array);

    /**
     * Constructor for map
     *
     * @param map  a (string, pnode)-map
     */
    pnode(const map_type& map);

    /**
     * bool conversion. It's true if and only if it is not empty.
     */
    operator bool() const noexcept;

    /**
     * Get the current node status.
     *
     * @return the status
     */
    status_t get_status() const;

    /**
     * Get the array. It will throw error if the current node does not hold an
     * array.
     *
     * @return the array const reference
     */
    const array_type& get_array() const;

    /**
     * Get the map. It will throw error if the current node does not hold an
     * map.
     *
     * @return the map const reference
     */
    const map_type& get_map() const;

    /**
     * Get the boolean value. It will throw error if the current node does not
     * hold an boolean value.
     *
     * @return the boolean value
     */
    bool get_boolean() const;

    /**
     * Get the integer value. It will throw error if the current node does not
     * hold an integer value.
     *
     * @return the integer value with type int64_t
     */
    std::int64_t get_integer() const;

    /**
     * Get the real floating point value. It will throw error if the current
     * node does not hold an real value.
     *
     * @return the real value with type double
     */
    double get_real() const;

    /**
     * Get the string. It will throw error if the current node does not hold an
     * string.
     *
     * @return the string
     */
    std::string get_string() const;

    /**
     * This function is to access the data under the map. It will throw error
     * when it does not hold a map. When access non-existent path in the map, it
     * will return an empty node.
     *
     * @param path  the key of the map
     *
     * @return node const reference. If the map does not have the path, return
     * an empty node.
     */
    const pnode& get(const std::string& path) const;

    /**
     * This function is to access the data under the array. It will throw error
     * when it does not hold an array or access out-of-bound index.
     *
     * @param index  the index for array
     *
     * @return node const reference.
     */
    const pnode& get(int index) const;

private:
    void throw_if_not_contain(status_t status) const;

    static const pnode& empty_node();

    status_t status_;
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


template <typename T,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
pnode::pnode(T integer) : status_(status_t::integer)
{
    if (integer > std::numeric_limits<std::int64_t>::max()) {
        throw std::runtime_error("The input is larger than int64_t.");
    }
    union_data_.integer_ = static_cast<std::int64_t>(integer);
}


}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_

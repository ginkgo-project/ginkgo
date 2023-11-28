// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_
#define GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_


#include <cassert>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


#include <ginkgo/core/config/data.hpp>


namespace gko {
namespace config {


/**
 * pnode is to describe the property tree
 */
class pnode {
public:
    using key_type = std::string;
    using data_type = data;

    enum class status_t { empty, array, data, map };

    pnode() : status_(status_t::empty) {}

    pnode(const data_type& d) : status_(status_t::data), data_(d) {}

    pnode(const std::vector<pnode>& array)
        : status_(status_t::array), array_(array)
    {}

    pnode(const std::map<key_type, pnode>& list)
        : status_(status_t::map), list_(list)
    {}

    // bool conversion. It's true if and only if it contains data.
    operator bool() const noexcept { return status_ != status_t::empty; }

    const data_type& get_data() const
    {
        this->throw_if_not_contain(status_t::data);
        return data_;
    }

    std::vector<pnode>& get_array()
    {
        this->throw_if_not_contain(status_t::array, true);
        status_ = status_t::array;
        return array_;
    }

    const std::vector<pnode>& get_array() const
    {
        this->throw_if_not_contain(status_t::array);
        return array_;
    }

    std::map<key_type, pnode>& get_map()
    {
        this->throw_if_not_contain(status_t::map, true);
        status_ = status_t::map;
        return list_;
    }

    const std::map<key_type, pnode>& get_map() const
    {
        this->throw_if_not_contain(status_t::map);
        return list_;
    }

    // Get the data of node's content
    template <typename T>
    T get_data() const
    {
        this->throw_if_not_contain(status_t::data);
        return gko::config::get<T>(data_);
    }

    pnode& at(const std::string& path)
    {
        this->throw_if_not_contain(status_t::map);
        return list_.at(path);
    }

    const pnode& at(const std::string& path) const
    {
        this->throw_if_not_contain(status_t::map);
        return list_.at(path);
    }

    // Return the object if it is found. Otherwise, return empty object.
    const pnode& get(const std::string& path) const
    {
        this->throw_if_not_contain(status_t::map);
        if (this->contains(path)) {
            return list_.at(path);
        } else {
            return pnode::empty_node();
        }
    }

    pnode& at(int i)
    {
        this->throw_if_not_contain(status_t::array);
        return array_.at(i);
    }

    const pnode& at(int i) const
    {
        this->throw_if_not_contain(status_t::array);
        return array_.at(i);
    }

    // Check the status
    bool is(status_t s) const { return this->get_status() == s; }

    bool contains(std::string key) const
    {
        this->throw_if_not_contain(status_t::map);
        auto it = list_.find(key);
        return (it != list_.end());
    }

    status_t get_status() const { return status_; }

protected:
    void throw_if_not_contain(status_t status, bool allow_empty = false) const;

    static const pnode& empty_node();

private:
    std::vector<pnode> array_;        // for array
    std::map<key_type, pnode> list_;  // for list
    data_type data_;                  // for value
    status_t status_;
};


}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_

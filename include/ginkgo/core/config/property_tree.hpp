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

    // Get the content of node
    data_type& get_data()
    {
        assert(status_ == status_t::data);
        return data_;
    }

    const data_type& get_data() const
    {
        assert(status_ == status_t::data);
        return data_;
    }

    std::vector<pnode>& get_array()
    {
        assert(status_ == status_t::array || status_ == status_t::empty);
        status_ = status_t::array;
        return array_;
    }

    const std::vector<pnode>& get_array() const
    {
        assert(status_ == status_t::array);
        return array_;
    }

    std::map<key_type, pnode>& get_map()
    {
        assert(status_ == status_t::map || status_ == status_t::empty);
        status_ = status_t::map;
        return list_;
    }

    const std::map<key_type, pnode>& get_map() const
    {
        assert(status_ == status_t::map);
        return list_;
    }

    // Get the data of node's content
    template <typename T>
    T get_data() const
    {
        assert(status_ == status_t::data);
        return gko::config::get<T>(data_);
    }

    pnode& at(const std::string& path)
    {
        assert(status_ == status_t::map);
        return list_.at(path);
    }

    const pnode& at(const std::string& path) const
    {
        assert(status_ == status_t::map);
        return list_.at(path);
    }

    // Return the objec if it is found. Otherwise, return empty object.
    const pnode& get(const std::string& path) const
    {
        assert(status_ == status_t::map);
        if (this->contains(path)) {
            return list_.at(path);
        } else {
            return pnode::empty_pn;
        }
    }

    pnode& at(int i)
    {
        assert(status_ == status_t::array);
        return array_.at(i);
    }

    const pnode& at(int i) const
    {
        assert(status_ == status_t::array);
        return array_.at(i);
    }

    // Check the status
    bool is(status_t s) const { return this->get_status() == s; }

    bool contains(std::string key) const
    {
        assert(status_ == status_t::map);
        auto it = list_.find(key);
        return (it != list_.end());
    }

    status_t get_status() const { return status_; }

private:
    std::vector<pnode> array_;        // for array
    std::map<key_type, pnode> list_;  // for list
    data_type data_;                  // for value
    status_t status_;
    const static pnode empty_pn;
};


}  // namespace config
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONFIG_PROPERTY_TREE_HPP_

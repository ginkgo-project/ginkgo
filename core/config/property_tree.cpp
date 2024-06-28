// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/config/property_tree.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace config {


pnode::pnode() : tag_(tag_t::empty) {}


pnode::pnode(bool boolean) : tag_(tag_t::boolean)
{
    union_data_.boolean_ = boolean;
}


pnode::pnode(const std::string& str) : tag_(tag_t::string) { str_ = str; }


pnode::pnode(double real) : tag_(tag_t::real) { union_data_.real_ = real; }


pnode::pnode(const char* str) : pnode(std::string(str)) {}


pnode::pnode(const array_type& array) : tag_(tag_t::array), array_(array) {}


pnode::pnode(const map_type& map) : tag_(tag_t::map), map_(map) {}


pnode::operator bool() const noexcept { return tag_ != tag_t::empty; }


pnode::tag_t pnode::get_tag() const { return tag_; }

const pnode::array_type& pnode::get_array() const
{
    this->throw_if_not_contain(tag_t::array);
    return array_;
}


const pnode::map_type& pnode::get_map() const
{
    this->throw_if_not_contain(tag_t::map);
    return map_;
}


bool pnode::get_boolean() const
{
    this->throw_if_not_contain(tag_t::boolean);
    return union_data_.boolean_;
}


std::int64_t pnode::get_integer() const
{
    this->throw_if_not_contain(tag_t::integer);
    return union_data_.integer_;
}


double pnode::get_real() const
{
    this->throw_if_not_contain(tag_t::real);
    return union_data_.real_;
}


const std::string& pnode::get_string() const
{
    this->throw_if_not_contain(tag_t::string);
    return str_;
}


const pnode& pnode::get(const std::string& key) const
{
    this->throw_if_not_contain(tag_t::map);
    auto it = map_.find(key);
    if (it != map_.end()) {
        return map_.at(key);
    } else {
        return pnode::empty_node();
    }
}

const pnode& pnode::get(int index) const
{
    this->throw_if_not_contain(tag_t::array);
    return array_.at(index);
}


void pnode::throw_if_not_contain(tag_t tag) const
{
    static auto str_tag = [](tag_t tag) -> std::string {
        if (tag == tag_t::empty) {
            return "empty";
        } else if (tag == tag_t::array) {
            return "array";
        } else if (tag == tag_t::map) {
            return "map";
        } else if (tag == tag_t::real) {
            return "real";
        } else if (tag == tag_t::boolean) {
            return "boolean";
        } else if (tag == tag_t::integer) {
            return "integer";
        } else if (tag == tag_t::string) {
            return "string";
        } else {
            return "unknown";
        }
    };
    bool is_valid = (tag_ == tag);
    std::string msg =
        "Contains " + str_tag(tag_) + ", but try to get " + str_tag(tag);
    GKO_THROW_IF_INVALID(is_valid, msg);
}


const pnode& pnode::empty_node()
{
    static pnode empty_pnode{};
    return empty_pnode;
}


}  // namespace config
}  // namespace gko

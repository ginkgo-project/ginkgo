// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/property_tree.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace config {


pnode::pnode() : status_(status_t::empty) {}


pnode::pnode(bool boolean) : status_(status_t::boolean)
{
    union_data_.boolean_ = boolean;
}


pnode::pnode(const std::string& str) : status_(status_t::string) { str_ = str; }


pnode::pnode(double real) : status_(status_t::real)
{
    union_data_.real_ = real;
}


pnode::pnode(const char* str) : pnode(std::string(str)) {}


pnode::pnode(const array_type& array) : status_(status_t::array), array_(array)
{}


pnode::pnode(const map_type& map) : status_(status_t::map), map_(map) {}


pnode::operator bool() const noexcept { return status_ != status_t::empty; }


pnode::status_t pnode::get_status() const { return status_; }

const pnode::array_type& pnode::get_array() const
{
    this->throw_if_not_contain(status_t::array);
    return array_;
}


const pnode::map_type& pnode::get_map() const
{
    this->throw_if_not_contain(status_t::map);
    return map_;
}


bool pnode::get_boolean() const
{
    this->throw_if_not_contain(status_t::boolean);
    return union_data_.boolean_;
}


std::int64_t pnode::get_integer() const
{
    this->throw_if_not_contain(status_t::integer);
    return union_data_.integer_;
}


double pnode::get_real() const
{
    this->throw_if_not_contain(status_t::real);
    return union_data_.real_;
}


std::string pnode::get_string() const
{
    this->throw_if_not_contain(status_t::string);
    return str_;
}


const pnode& pnode::get(const std::string& path) const
{
    this->throw_if_not_contain(status_t::map);
    auto it = map_.find(path);
    if (it != map_.end()) {
        return map_.at(path);
    } else {
        return pnode::empty_node();
    }
}

const pnode& pnode::get(int index) const
{
    this->throw_if_not_contain(status_t::array);
    return array_.at(index);
}


void pnode::throw_if_not_contain(status_t status) const
{
    static auto str_status = [](status_t status) -> std::string {
        if (status == status_t::empty) {
            return "empty";
        } else if (status == status_t::array) {
            return "array";
        } else if (status == status_t::map) {
            return "map";
        } else if (status == status_t::real) {
            return "real";
        } else if (status == status_t::boolean) {
            return "boolean";
        } else if (status == status_t::integer) {
            return "integer";
        } else if (status == status_t::string) {
            return "string";
        } else {
            return "unknown";
        }
    };
    bool is_valid = (status_ == status);
    std::string msg = "Contains " + str_status(status_) + ", but try to get " +
                      str_status(status);
    GKO_THROW_IF_INVALID(is_valid, msg);
}


const pnode& pnode::empty_node()
{
    static pnode empty_pn{};
    return empty_pn;
}


}  // namespace config
}  // namespace gko

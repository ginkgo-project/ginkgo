// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/property_tree.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace config {


void pnode::throw_if_not_contain(status_t status, bool allow_empty) const
{
    static auto str_status = [](status_t status) -> std::string {
        if (status == status_t::empty) {
            return "empty";
        } else if (status == status_t::array) {
            return "array";
        } else if (status == status_t::map) {
            return "map";
        } else if (status == status_t::data) {
            return "data";
        } else {
            return "unknown";
        }
    };
    bool is_valid =
        (status_ == status || (allow_empty && status_ == status_t::empty));
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

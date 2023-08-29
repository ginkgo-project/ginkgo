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

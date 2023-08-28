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

#ifndef GKO_CORE_TEST_CONFIG_UTILS_HPP_
#define GKO_CORE_TEST_CONFIG_UTILS_HPP_


#include <ostream>


#include <ginkgo/core/config/data.hpp>
#include <ginkgo/core/config/property_tree.hpp>

namespace gko {
namespace config {

std::ostream& operator<<(std::ostream& stream, const data& d)
{
    if (mpark::holds_alternative<std::string>(d)) {
        stream << '"' << mpark::get<std::string>(d) << '"';
    } else if (mpark::holds_alternative<long long int>(d)) {
        stream << mpark::get<long long int>(d);
    } else if (mpark::holds_alternative<double>(d)) {
        stream << mpark::get<double>(d);
    } else if (mpark::holds_alternative<bool>(d)) {
        stream << (mpark::get<bool>(d) ? "true" : "false");
    } else if (mpark::holds_alternative<mpark::monostate>(d)) {
        stream << "<empty>";
    }
    return stream;
}

// For debug usage
void print(std::ostream& stream, const pnode& tree, int offset = 0)
{
    std::string offset_str(offset, ' ');
    if (tree.is(pnode::status_t::array)) {
        stream << "[" << std::endl;
        for (const auto node : tree.get_array()) {
            stream << offset_str << "  ";
            print(stream, node, offset + 2);
        }
        stream << offset_str << "]" << std::endl;
    } else if (tree.is(pnode::status_t::map)) {
        stream << "{" << std::endl;
        for (const auto node : tree.get_map()) {
            stream << offset_str << "  " << node.first << ": ";
            print(stream, node.second, offset + 2);
        }
        stream << offset_str << "}" << std::endl;
    } else if (tree.is(pnode::status_t::data)) {
        stream << tree.get_data() << std::endl;
    } else {
        stream << "empty_node" << std::endl;
    }
}

}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_TEST_CONFIG_UTILS_HPP_

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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TEMPLATE_HELPER_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TEMPLATE_HELPER_HPP_


#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>


namespace gko {
namespace extensions {
namespace file_config {


// return the input string without space
inline std::string remove_space(const std::string& str)
{
    std::string nospace = str;
    nospace.erase(remove_if(nospace.begin(), nospace.end(),
                            [](char x) { return x == ' '; }),
                  nospace.end());
    return nospace;
}


// get the base class of input
inline std::string get_base_class(const std::string& str)
{
    auto langle_pos = str.find("<");
    return remove_space(str.substr(0, langle_pos));
}


// get the template string in the first pair of <>
inline std::string get_base_template(const std::string& str)
{
    auto langle_pos = str.find("<");
    auto rangle_pos = str.rfind(">");
    if (rangle_pos > langle_pos && rangle_pos != std::string::npos) {
        return remove_space(
            str.substr(langle_pos + 1, rangle_pos - langle_pos - 1));
    } else {
        return "";
    }
}


//  find the position of separator `,` of input string
inline std::size_t find_template_sep(const std::string& str,
                                     std::size_t pos = 0)
{
    int is_closed = 0;
    for (pos; pos < str.length(); pos++) {
        is_closed += str[pos] == '<';
        is_closed -= str[pos] == '>';
        if (is_closed == 0 && str[pos] == ',') {
            return pos;
        }
    }
    assert(is_closed == 0);
    return pos;
}


// Base on the base_template and type_template to decide the final template
// string. The base_template has higher priority than type_template.
inline std::string combine_template(const std::string& base_template,
                                    const std::string& type_template)
{
    // base_template and type_template must not contain space
    std::string combined;
    std::size_t curr_base_pos = 0;
    std::size_t curr_type_pos = 0;
    std::size_t index = 0;
    while (curr_type_pos < type_template.length() ||
           curr_base_pos < base_template.length()) {
        // move a char when it is not the first one
        auto base_sep = find_template_sep(base_template,
                                          curr_base_pos + (curr_base_pos != 0));
        auto type_sep = find_template_sep(type_template,
                                          curr_type_pos + (curr_type_pos != 0));
        if (base_sep > curr_base_pos + 1) {
            // base contain the information, use it
            combined +=
                base_template.substr(curr_base_pos, base_sep - curr_base_pos);
        } else {
            // use the default or the setting from type
            combined +=
                type_template.substr(curr_type_pos, type_sep - curr_type_pos);
        }
        if (base_sep <= curr_base_pos + (curr_base_pos != 0) &&
            type_sep <= curr_type_pos + (curr_type_pos != 0)) {
            // both are empty
            std::cerr << "The " << index
                      << "-th (0-based) template parameter is empty"
                      << std::endl;
            assert(false);
        }
        curr_base_pos = base_sep;
        curr_type_pos = type_sep;
        index++;
    }
    return combined;
}


}  // namespace file_config
}  // namespace extensions
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TEMPLATE_HELPER_HPP_

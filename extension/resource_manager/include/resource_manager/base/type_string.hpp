/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_TYPE_STRING_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_TYPE_STRING_HPP_


#include <string>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "resource_manager/base/type_pack.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


/**
 * GET_STRING_PARTIAL is to generate a specialization of get_string, which gives
 * `get_string<_type> = _str`.
 *
 * @param _type  the type
 * @param _str  the corresponding string
 */
#define GET_STRING_PARTIAL(_type, _str)                                      \
    template <>                                                              \
    std::string get_string<_type>()                                          \
    {                                                                        \
        return _str;                                                         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

/**
 * get_string returns the string identifier of type
 *
 * @tparam T  the type
 *
 * @return the identifier string
 *
 * @note the identifier string must be identical among this system
 */
template <typename T>
std::string get_string();

GET_STRING_PARTIAL(double, "double");
GET_STRING_PARTIAL(float, "float");
GET_STRING_PARTIAL(gko::int32, "int");
GET_STRING_PARTIAL(gko::int64, "int64");
GET_STRING_PARTIAL(gko::uint32, "uint32");
using isai_lower =
    std::integral_constant<gko::preconditioner::isai_type,
                           gko::preconditioner::isai_type::lower>;
using isai_upper =
    std::integral_constant<gko::preconditioner::isai_type,
                           gko::preconditioner::isai_type::upper>;
using isai_general =
    std::integral_constant<gko::preconditioner::isai_type,
                           gko::preconditioner::isai_type::general>;
using isai_spd = std::integral_constant<gko::preconditioner::isai_type,
                                        gko::preconditioner::isai_type::spd>;
GET_STRING_PARTIAL(isai_lower, "isai_lower");
GET_STRING_PARTIAL(isai_upper, "isai_upper");
GET_STRING_PARTIAL(isai_general, "isai_general");
GET_STRING_PARTIAL(isai_spd, "isai_spd");
GET_STRING_PARTIAL(gko::solver::LowerTrs<>, "LowerTrs");
GET_STRING_PARTIAL(gko::solver::UpperTrs<>, "UpperTrs");
GET_STRING_PARTIAL(std::true_type, "true");
GET_STRING_PARTIAL(std::false_type, "false");
GET_STRING_PARTIAL(bool, "bool");


/**
 * @copydoc get_string<T>()
 *
 * @param T the type input
 *
 * @note this is another version such that allow arg input.
 */
template <typename T>
std::string get_string(T)
{
    return get_string<T>();
}

/**
 * get_string for the type_list ending case. it will return the item's
 * identifier.
 *
 * @tparam K  the type in the type_list of input
 *
 * @param type_list<K>  the type_list input
 */
template <typename K>
std::string get_string(type_list<K>)
{
    return get_string<K>();
}

/**
 * get_string for the type_list general case. it will return the first item's
 * identifier + the Rest items' identidier with `+` separator.
 *
 * @tparam K  the first type in the type_list of input
 * @tparam ...Rest  the rest types in the type_list of input
 *
 * @param type_list<K, Rest...>  the type_list input
 */
template <typename K, typename... Rest>
typename std::enable_if<(sizeof...(Rest) > 0), std::string>::type get_string(
    type_list<K, Rest...>)
{
    return get_string<K>() + "+" + get_string(type_list<Rest...>());
}


std::string create_type_name(const std::string& arg) { return arg; }

template <typename... Rest>
std::string create_type_name(const std::string& arg, Rest&&... rest)
{
    return arg + "+" + create_type_name(std::forward<Rest>(rest)...);
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko

#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_TYPE_STRING_HPP_

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

#ifndef GKO_BENCHMARK_UTILS_JSON_HPP_
#define GKO_BENCHMARK_UTILS_JSON_HPP_


#include <ginkgo/ginkgo.hpp>


#include <type_traits>


#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>


// helper for setting rapidjson object members
template <typename T, typename NameType, typename Allocator>
std::enable_if_t<
    !std::is_same<typename std::decay<T>::type, gko::size_type>::value, void>
add_or_set_member(rapidjson::Value& object, NameType&& name, T&& value,
                  Allocator&& allocator)
{
    if (object.HasMember(name)) {
        object[name] = std::forward<T>(value);
    } else {
        auto n = rapidjson::Value(name, allocator);
        object.AddMember(n, std::forward<T>(value), allocator);
    }
}


/**
   @internal This is required to fix some MacOS problems (and possibly other
   compilers). There is no explicit RapidJSON constructor for `std::size_t` so a
   conversion to a known constructor is required to solve any ambiguity. See the
   last comments of https://github.com/ginkgo-project/ginkgo/issues/270.
 */
template <typename T, typename NameType, typename Allocator>
std::enable_if_t<
    std::is_same<typename std::decay<T>::type, gko::size_type>::value, void>
add_or_set_member(rapidjson::Value& object, NameType&& name, T&& value,
                  Allocator&& allocator)
{
    if (object.HasMember(name)) {
        object[name] =
            std::forward<std::uint64_t>(static_cast<std::uint64_t>(value));
    } else {
        auto n = rapidjson::Value(name, allocator);
        object.AddMember(
            n, std::forward<std::uint64_t>(static_cast<std::uint64_t>(value)),
            allocator);
    }
}


// helper for writing out rapidjson Values
inline std::ostream& operator<<(std::ostream& os, const rapidjson::Value& value)
{
    rapidjson::OStreamWrapper jos(os);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper, rapidjson::UTF8<>,
                            rapidjson::UTF8<>, rapidjson::CrtAllocator,
                            rapidjson::kWriteNanAndInfFlag>
        writer(jos);
    value.Accept(writer);
    return os;
}


#endif  // GKO_BENCHMARK_UTILS_JSON_HPP_

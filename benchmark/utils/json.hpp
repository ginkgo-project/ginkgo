// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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

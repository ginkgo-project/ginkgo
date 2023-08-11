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

#ifndef GINKGO_CONFIG_HPP
#define GINKGO_CONFIG_HPP


#include <ginkgo/config.hpp>

#include <map>
#include <variant>


namespace gko {


struct Configurator;


struct property_tree {
    template <typename T>
    std::optional<T> get(std::string name) const
    {
        return {};
    }

    std::string name;
    std::string value;
};

struct context {
    std::map<std::string, std::shared_ptr<Configurator>> custom_map;
};


struct type_config {
    std::variant<double, float> value_type = double{};
    std::variant<int32, int64> index_type = int32{};
    std::variant<int32, int64> global_index_type = int64{};
};

template <typename ValueType, typename IndexType, typename GlobalIndexType>
struct compile_type_config {
    using value_type = ValueType;
    using index_type = IndexType;
    using global_index_type = GlobalIndexType;
};


template <typename DefaultType>
std::variant<float, double> encode_value_type(const property_tree& pt)
{
    if (pt.name != "global_index_type") {
        return DefaultType{};
    }

    if (pt.value == "float") {
        return float{};
    }
    if (pt.value == "double") {
        return double{};
    }

    throw std::runtime_error("unsupported value type");
}

template <typename DefaultType>
std::variant<int32, int64> encode_index_type(const property_tree& pt)
{
    if (pt.name != "global_index_type") {
        return DefaultType{};
    }

    if (pt.value == "int32") {
        return int32{};
    }
    if (pt.value == "int64") {
        return int64{};
    }

    throw std::runtime_error("unsupported value type");
}


template <typename ValueType, typename IndexType, typename GlobalIndexType>
type_config encode_type_config(const property_tree& pt)
{
    return {encode_value_type<ValueType>(pt), encode_index_type<IndexType>(pt),
            encode_index_type<GlobalIndexType>(pt)};
}


template <typename T, typename ValueType, typename IndexType,
          typename GlobalIndexType>
auto dispatch(const property_tree& pt, const context& ctx,
              const type_config& cfg)
{
    return T::template configure<ValueType, IndexType, GlobalIndexType>(pt,
                                                                        ctx);
}


template <typename T, typename... Types>
auto visitor(const property_tree& pt, const context& ctx,
             const type_config& cfg)
{
    return [&](auto var) {
        using type = std::decay_t<decltype(var)>;
        return dispatch<T, Types..., type>(pt, ctx, cfg);
    };
}


template <typename T, typename ValueType, typename IndexType>
auto dispatch(const property_tree& pt, const context& ctx,
              const type_config& cfg)
{
    return std::visit(visitor<T, ValueType, IndexType>(pt, ctx, cfg),
                      cfg.global_index_type);
}

template <typename T, typename ValueType>
auto dispatch(const property_tree& pt, const context& ctx,
              const type_config& cfg)
{
    return std::visit(visitor<T, ValueType>(pt, ctx, cfg), cfg.index_type);
}

template <typename T>
auto dispatch(const property_tree& pt, const context& ctx,
              const type_config& cfg)
{
    return std::visit(visitor<T>(pt, ctx, cfg), cfg.value_type);
}


using configure_fn = std::function<std::shared_ptr<LinOpFactory>(
    const property_tree&, const context&, const type_config&)>;


template <typename T>
configure_fn create_default_configure_fn()
{
    return [](const property_tree& pt, const context& ctx,
              const type_config& cfg) { return dispatch<T>(pt, ctx, cfg); };
}

namespace config {
struct Cg {
    template <typename ValueType, typename IndexType, typename GlobalIndexType>
    static std::shared_ptr<LinOpFactory> configure(const property_tree& pt,
                                                   const context& ctx)
    {
        // actual implementation
        return nullptr;
    }
};

enum isai_type {};
struct Isai {
    template <typename ValueType, typename IndexType, typename GlobalIndexType,
              isai_type IsaiType>
    static std::shared_ptr<LinOpFactory> configure(const property_tree& pt,
                                                   const context& ctx)
    {
        // actual implementation
        return nullptr;
    }

    template <typename ValueType, typename IndexType, typename GlobalIndexType>
    static std::shared_ptr<LinOpFactory> configure(const property_tree& pt,
                                                   const context& ctx)
    {
        // dispatch isai_type
        // actual implementation
        return nullptr;
    }
};
}  // namespace config


std::shared_ptr<LinOpFactory> parse(const property_tree& pt, const context& ctx,
                                    const type_config& tcfg = {})
{
    std::map<std::string, configure_fn> configurator_map = {
        {"cg", create_default_configure_fn<config::Cg>()}};

    return configurator_map[pt.value](pt, ctx, tcfg);
}

template <typename ValueType = default_precision, typename IndexType = int32,
          typename GlobalIndexType = int64>
std::shared_ptr<LinOpFactory> parse(const property_tree& pt, const context& ctx)
{
    auto type_config =
        encode_type_config<ValueType, IndexType, GlobalIndexType>(pt);
    return parse(pt, ctx, type_config);
}


template <typename T>
struct parse_helper {
    template <typename ValueType, typename IndexType, typename GlobalIndexType,
              typename Index>
    static T apply(Index index, const property_tree& pt, const context&)
    {
        pt.get<T>(index).value();
    }
};

template <typename T>
struct parse_helper<std::vector<T>> {
    template <typename ValueType, typename IndexType, typename GlobalIndexType,
              typename Index>
    static std::vector<T> apply(Index index, const property_tree& pt,
                                const context& ctx)
    {
        std::vector<T> result;
        auto properties = pt.get<std::vector<property_tree>>(index);
        for (int i = 0; i < properties.size(); ++i) {
            result.emplace_back(
                parse_helper<T>::template apply<ValueType, IndexType,
                                                GlobalIndexType>(i, pt, ctx));
        }
        return result;
    }
};


template <>
struct parse_helper<std::shared_ptr<LinOpFactory>> {
    template <typename ValueType, typename IndexType, typename GlobalIndexType,
              typename Index>
    static std::shared_ptr<LinOpFactory> apply(Index index,
                                               const property_tree& pt,
                                               const context& ctx)
    {
        return parse(pt.get<property_tree>(index).value(), ctx);
    }
};

template <>
struct parse_helper<std::shared_ptr<LinOp>> {
    template <typename ValueType, typename IndexType, typename GlobalIndexType,
              typename Index>
    static std::shared_ptr<LinOp> apply(Index index, const property_tree& pt,
                                        const context& ctx)
    {
        return ctx.custom_map.at(index);
    }
};

}  // namespace gko


#endif  // GINKGO_CONFIG_HPP

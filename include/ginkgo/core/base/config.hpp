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
    std::variant<double, float> value_type;
    std::variant<int, long> index_type;
    std::variant<int, long> global_index_type;
};

template <typename ValueType, typename IndexType, typename GlobalIndexType>
struct compile_type_config {
    using value_type = ValueType;
    using index_type = IndexType;
    using global_index_type = GlobalIndexType;
};

template <typename T>
struct concrete_type {
    using type = T;
};


template <typename DefaultValueType, typename DefaultIndexType,
          typename DefaultGlobalIndexType>
struct encode_type_config {
    static type_config apply(const property_tree& cfg)
    {
        if (cfg.name != "value_type") {
            return encode_type_config<concrete_type<DefaultValueType>,
                                      DefaultIndexType,
                                      DefaultGlobalIndexType>::apply(cfg);
        }

        if (cfg.value == "double") {
            return encode_type_config<concrete_type<double>, DefaultIndexType,
                                      DefaultGlobalIndexType>::apply(cfg);
        }
        if (cfg.value == "float") {
            return encode_type_config<concrete_type<float>, DefaultIndexType,
                                      DefaultGlobalIndexType>::apply(cfg);
        }

        throw std::runtime_error("unsupported value type");
    }
};

template <typename ValueType, typename DefaultIndexType,
          typename DefaultGlobalIndexType>
struct encode_type_config<concrete_type<ValueType>, DefaultIndexType,
                          DefaultGlobalIndexType> {
    static type_config apply(const property_tree& cfg)
    {
        if (cfg.name != "index_type") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<DefaultIndexType>,
                                      DefaultGlobalIndexType>::apply(cfg);
        }

        if (cfg.value == "int32") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<int32>,
                                      DefaultGlobalIndexType>::apply(cfg);
        }
        if (cfg.value == "int64") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<int64>,
                                      DefaultGlobalIndexType>::apply(cfg);
        }

        throw std::runtime_error("unsupported index type");
    }
};

template <typename ValueType, typename IndexType,
          typename DefaultGlobalIndexType>
struct encode_type_config<concrete_type<ValueType>, concrete_type<IndexType>,
                          DefaultGlobalIndexType> {
    static type_config apply(const property_tree& cfg)
    {
        if (cfg.name != "global_index_type") {
            return encode_type_config<
                concrete_type<ValueType>, concrete_type<IndexType>,
                concrete_type<DefaultGlobalIndexType>>::apply(cfg);
        }

        if (cfg.value == "int32") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<IndexType>,
                                      concrete_type<int32>>::apply(cfg);
        }
        if (cfg.value == "int64") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<IndexType>,
                                      concrete_type<int32>>::apply(cfg);
        }

        throw std::runtime_error("unsupported index type");
    }
};

template <typename ValueType, typename IndexType, typename GlobalIndexType>
struct encode_type_config<concrete_type<ValueType>, concrete_type<IndexType>,
                          concrete_type<GlobalIndexType>> {
    static type_config apply(const property_tree&)
    {
        return {ValueType{}, IndexType{}, GlobalIndexType{}};
    }
};


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


struct Configurator {
    virtual std::shared_ptr<LinOpFactory> configure(const property_tree& pt,
                                                    const context& ctx,
                                                    const type_config& cfg) = 0;
};

template <typename T>
struct EnableConfigurator : Configurator {
    std::shared_ptr<LinOpFactory> configure(const property_tree& pt,
                                            const context& ctx,
                                            const type_config& cfg) override
    {
        return dispatch<T>(pt, ctx, cfg);
    }
};


namespace config {
struct Cg : EnableConfigurator<Cg> {
    template <typename ValueType = double, typename IndexType = int,
              typename GlobalIndexType = long>
    static std::shared_ptr<LinOpFactory> configure(const property_tree& pt,
                                                   const context& ctx)
    {
        // actual implementation
        return nullptr;
    }
};
}  // namespace config


template <typename ValueType = default_precision, typename IndexType = int32,
          typename GlobalIndexType = int64>
std::shared_ptr<LinOpFactory> parse(const property_tree& pt, const context& ctx)
{
    std::map<std::string, std::shared_ptr<Configurator>> configurator_map = {
        {"cg", std::make_shared<config::Cg>()}};

    auto type_config =
        encode_type_config<ValueType, IndexType, GlobalIndexType>::apply(pt);

    return configurator_map[pt.value]->configure(pt, ctx, type_config);
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

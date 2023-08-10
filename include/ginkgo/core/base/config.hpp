#ifndef GINKGO_CONFIG_HPP
#define GINKGO_CONFIG_HPP


#include <ginkgo/config.hpp>

#include <map>
#include <variant>


namespace gko {


struct Configurator;


struct property_tree {
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

template <typename T>
struct concrete_type {
    using type = T;
};


template <typename DefaultValueType, typename DefaultIndexType,
          typename DefaultGlobalIndexType>
struct encode_type_config {
    type_config apply(const property_tree& cfg)
    {
        if (cfg.name != "value_type") {
            return encode_type_config<concrete_type<DefaultValueType>,
                                      DefaultIndexType, DefaultGlobalIndexType>(
                cfg);
        }

        if (cfg.value == "double") {
            return encode_type_config<concrete_type<double>, DefaultIndexType,
                                      DefaultGlobalIndexType>(cfg);
        }
        if (cfg.value == "float") {
            return encode_type_config<concrete_type<float>, DefaultIndexType,
                                      DefaultGlobalIndexType>(cfg);
        }

        throw std::runtime_error("unsupported value type");
    }
};

template <typename ValueType, typename DefaultIndexType,
          typename DefaultGlobalIndexType>
struct encode_type_config<concrete_type<ValueType>, DefaultIndexType,
                          DefaultGlobalIndexType> {
    type_config apply(const property_tree& cfg)
    {
        if (cfg.name != "index_type") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<DefaultIndexType>,
                                      DefaultGlobalIndexType>(cfg);
        }

        if (cfg.value == "int32") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<int32>,
                                      DefaultGlobalIndexType>(cfg);
        }
        if (cfg.value == "int64") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<int64>,
                                      DefaultGlobalIndexType>(cfg);
        }

        throw std::runtime_error("unsupported index type");
    }
};

template <typename ValueType, typename IndexType,
          typename DefaultGlobalIndexType>
struct encode_type_config<concrete_type<ValueType>, concrete_type<IndexType>,
                          DefaultGlobalIndexType> {
    type_config apply(const property_tree& cfg)
    {
        if (cfg.name != "global_index_type") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<IndexType>,
                                      concrete_type<DefaultGlobalIndexType>>(
                cfg);
        }

        if (cfg.value == "int32") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<IndexType>,
                                      concrete_type<int32>>(cfg);
        }
        if (cfg.value == "int64") {
            return encode_type_config<concrete_type<ValueType>,
                                      concrete_type<IndexType>,
                                      concrete_type<int32>>(cfg);
        }

        throw std::runtime_error("unsupported index type");
    }
};

template <typename ValueType, typename IndexType, typename GlobalIndexType>
struct encode_type_config<concrete_type<ValueType>, concrete_type<IndexType>,
                          concrete_type<GlobalIndexType>> {
    type_config apply(const property_tree&)
    {
        return {ValueType{}, IndexType{}, GlobalIndexType{}};
    }
};


template <typename T, typename ValueType, typename IndexType>
auto dispatch(const property_tree& pt, const context& ctx,
              const type_config& cfg)
{
    if (std::holds_alternative<int>(cfg.index_type)) {
        return T::template configure<ValueType, IndexType, int>(pt, ctx);
    }
    if (std::holds_alternative<long>(cfg.index_type)) {
        return T::template configure<ValueType, IndexType, int>(pt, ctx);
    }

    throw std::runtime_error("unsupported global index type");
}


template <typename T, typename ValueType>
auto dispatch(const property_tree& pt, const context& ctx,
              const type_config& cfg)
{
    if (std::holds_alternative<int>(cfg.index_type)) {
        return dispatch<T, double, int>(pt, ctx, cfg);
    }
    if (std::holds_alternative<long>(cfg.index_type)) {
        return dispatch<T, double, long>(pt, ctx, cfg);
    }

    throw std::runtime_error("unsupported value type");
}


template <typename T>
auto dispatch(const property_tree& pt, const context& ctx,
              const type_config& cfg)
{
    if (std::holds_alternative<double>(cfg.value_type)) {
        return dispatch<T, double>(pt, ctx, cfg);
    }
    if (std::holds_alternative<float>(cfg.value_type)) {
        return dispatch<T, float>(pt, ctx, cfg);
    }

    throw std::runtime_error("unsupported value type");
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


}  // namespace gko


#endif  // GINKGO_CONFIG_HPP

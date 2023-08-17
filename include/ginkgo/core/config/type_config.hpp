//
// Created by marcel on 17.08.23.
//

#ifndef GINKGO_TYPE_CONFIG_HPP
#define GINKGO_TYPE_CONFIG_HPP

#include <ginkgo/config.hpp>

#include <string>
#include <variant>

#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace config {


struct type_config {
    std::variant<float, double> value_type = double{};
    std::variant<int32, int64> index_type = int32{};
    std::variant<int32, int64> global_index_type = int64{};
};


inline type_config default_type_config()
{
    return {double{}, int32{}, int64{}};
}


template <typename T>
struct alternative {
    alternative(std::string name_, T) : name(std::move(name_)) {}

    std::string name;
    [[nodiscard]] T get() const { return {}; }
};


template <typename... Alternatives>
std::variant<Alternatives...> encode_type(
    const property_tree& pt, std::variant<Alternatives...> default_value,
    const std::string& name, const alternative<Alternatives>&... alternatives)
{
    if (!pt.contains(name)) {
        return default_value;
    }

    std::variant<Alternatives...> return_variant{};

    bool found_alternative = false;
    auto fill_variant = [&](const auto& alt) {
        if (pt.at(name).template get_data<std::string>() == alt.name) {
            return_variant = alt.get();
            found_alternative = true;
        }
    };
    (fill_variant(alternatives), ...);

    // maybe check for multiple alternatives with same name
    if (!found_alternative) {
        throw std::runtime_error("unsupported type for " + name);
    }

    return return_variant;
}


inline type_config encode_type_config(
    const property_tree& pt,
    const type_config& default_cfg = default_type_config())
{
    type_config tc;
    tc.value_type = encode_type(pt, default_cfg.value_type, "value_type",
                                alternative{"float32", float{}},
                                alternative{"float64", double{}});
    tc.index_type = encode_type(pt, default_cfg.index_type, "index_type",
                                alternative{"int32", int32{}},
                                alternative{"int64", int64{}});
    tc.global_index_type = encode_type(
        pt, default_cfg.global_index_type, "global_index_type",
        alternative{"int32", int32{}}, alternative{"int64", int64{}});
    return tc;
}


}  // namespace config
}  // namespace gko

#endif  // GINKGO_TYPE_CONFIG_HPP

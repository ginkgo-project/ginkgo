#include <ginkgo/config/parser.hpp>
#include <json.hpp>
#include "ginkgo/core/base/types.hpp"
#include "ginkgo/core/preconditioner/jacobi.hpp"
#include "ginkgo/core/solver/cg.hpp"
#include "ginkgo/core/solver/fcg.hpp"
#include "ginkgo/core/solver/gmres.hpp"
#include "ginkgo/core/stop/iteration.hpp"
#include "ginkgo/core/stop/residual_norm.hpp"
#include "ginkgo/core/stop/time.hpp"

namespace gko {
namespace config {


template <typename ValueType, typename IndexType>
deferred_factory_parameter<LinOpFactory> parse_linop_config(
    const nlohmann::json& object);


template <typename ParametersType>
struct additional_parameters {
    void parse(const nlohmann::json& object, ParametersType& params) {}
};


#define PARSE_PARAM(_json_name, _type, _param_name)    \
    if (object.contains(#_json_name)) {                \
        auto value = object[#_json_name].get<_type>(); \
        params.with_##_param_name(value);              \
    }


template <typename ValueType>
struct additional_parameters<solver::Gmres<ValueType>> {
    void parse(const nlohmann::json& object,
               typename solver::Gmres<ValueType>::parameters_type& params)
    {
        PARSE_PARAM(restart, unsigned, krylov_dim);
        PARSE_PARAM(flexible, bool, flexible);
    }
};


template <typename ValueType, typename IndexType, typename ParametersType>
void parse_iterative_solver_parameters_type(const nlohmann::json& object,
                                            ParametersType& params)
{
    GKO_ASSERT(object.contains("stop") && object["stop"].is_object());
    for (auto pair : object["stop"].items()) {
        if (pair.key() == "time") {
            auto time = pair.value().get<int>();
            params.criterion_generators.emplace_back(
                gko::stop::Time::build().with_time_limit(
                    std::chrono::milliseconds{time}));
        } else if (pair.key() == "iterations") {
            auto count = pair.value().get<unsigned>();
            params.criterion_generators.emplace_back(
                gko::stop::Iteration::build().with_max_iters(count));
        } else if (pair.key() == "rel_residual") {
            auto res = pair.value().get<remove_complex<ValueType>>();
            params.criterion_generators.emplace_back(
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_baseline(gko::stop::mode::rhs_norm)
                    .with_reduction_factor(res));
        } else if (pair.key() == "abs_residual") {
            auto res = pair.value().template get<remove_complex<ValueType>>();
            params.criterion_generators.emplace_back(
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_baseline(gko::stop::mode::absolute)
                    .with_reduction_factor(res));
        } else if (pair.key() == "residual_reduction") {
            auto res = pair.value().template get<remove_complex<ValueType>>();
            params.criterion_generators.emplace_back(
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_baseline(gko::stop::mode::initial_resnorm)
                    .with_reduction_factor(res));
        } else {
            throw std::runtime_error("Unknown stopping criterion type");
        }
    }
}


template <typename ValueType, typename IndexType, typename ParametersType>
void parse_preconditioned_iterative_solver_parameters_type(
    const nlohmann::json& object, ParametersType& params)
{
    parse_iterative_solver_parameters_type<ValueType, IndexType>(object,
                                                                 params);
    if (object.contains("preconditioner")) {
        GKO_ASSERT(object["preconditioner"].is_object());
        params.preconditioner_generator =
            parse_linop_config<ValueType, IndexType>(object["preconditioner"]);
    }
}


#define REGISTER_PRECONDITIONED_ITERATIVE_TYPE(_typename) \
    {                                                     \
#_typename, [](const nlohmann::json& object) { \
auto params = _typename<ValueType>::build(); \
parse_preconditioned_iterative_solver_parameters_type<ValueType, IndexType>(object, params); \
 return params; }          \
    }


template <typename ValueType, typename IndexType>
deferred_factory_parameter<LinOpFactory> parse_linop_config(
    const nlohmann::json& object)
{
    std::map<std::string,
             std::function<deferred_factory_parameter<LinOpFactory>(
                 const nlohmann::json&)>>
        object_map{
            REGISTER_PRECONDITIONED_ITERATIVE_TYPE(solver::Cg),
            REGISTER_PRECONDITIONED_ITERATIVE_TYPE(solver::Fcg),
            REGISTER_PRECONDITIONED_ITERATIVE_TYPE(solver::Gmres),
            {"preconditioner::Jacobi", [](const nlohmann::json& object) {
                 auto params =
                     preconditioner::Jacobi<ValueType, IndexType>::build();
                 PARSE_PARAM(max_block_size, unsigned, max_block_size);
                 return params;
             }}};
    return object_map.at(object["type"].get<std::string>())(object);
}


template <typename ValueType, typename IndexType>
std::shared_ptr<const LinOpFactory> parse_config(
    std::istream& stream, std::shared_ptr<const Executor> exec)
{
    auto object = nlohmann::json::parse(stream);
    return parse_linop_config<ValueType, IndexType>(object).on(exec);
}


#define GKO_DECLARE_PARSE_CONFIG(ValueType, IndexType)                      \
    std::shared_ptr<const LinOpFactory> parse_config<ValueType, IndexType>( \
        std::istream & stream, std::shared_ptr<const Executor> exec)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PARSE_CONFIG);


}  // namespace config
}  // namespace gko
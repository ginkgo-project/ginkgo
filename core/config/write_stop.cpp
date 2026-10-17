// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <string>

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/config/write_helper.hpp"


namespace gko {
namespace config {


// The writers mirror the criterion parsers in core/config/stop_config.cpp.


pnode parameter_writer<stop::Combined>::write(
    const stop::Combined::parameters_type& params)
{
    return write_criteria(params.criteria);
}


pnode parameter_writer<stop::Iteration>::write(
    const stop::Iteration::parameters_type& params)
{
    auto map = make_config_map("Iteration");
    map["max_iters"] = write_value(params.max_iters);
    return pnode{map};
}


pnode parameter_writer<stop::Time>::write(
    const stop::Time::parameters_type& params)
{
    auto map = make_config_map("Time");
    // configure_time reads the limit as a count of nanoseconds
    map["time_limit"] = write_value(params.time_limit.count());
    return pnode{map};
}


namespace {


// ResidualNorm and ImplicitResidualNorm have the same parameters
template <typename ValueType, typename ParametersType>
pnode write_residual_norm(const std::string& type, const ParametersType& params)
{
    auto map = make_config_map<ValueType>(type);
    map["reduction_factor"] = write_value(params.reduction_factor);
    map["baseline"] = write_value(params.baseline);
    return pnode{map};
}


}  // namespace


template <typename ValueType>
pnode parameter_writer<stop::ResidualNorm<ValueType>>::write(
    const typename stop::ResidualNorm<ValueType>::parameters_type& params)
{
    return write_residual_norm<ValueType>("ResidualNorm", params);
}

#define GKO_DECLARE_RESIDUAL_NORM_WRITER(ValueType) \
    struct parameter_writer<stop::ResidualNorm<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_WRITER);


template <typename ValueType>
pnode parameter_writer<stop::ImplicitResidualNorm<ValueType>>::write(
    const typename stop::ImplicitResidualNorm<ValueType>::parameters_type&
        params)
{
    return write_residual_norm<ValueType>("ImplicitResidualNorm", params);
}

#define GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_WRITER(ValueType) \
    struct parameter_writer<stop::ImplicitResidualNorm<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_WRITER);


}  // namespace config
}  // namespace gko

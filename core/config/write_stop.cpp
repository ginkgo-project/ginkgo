// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/config/write_helper.hpp"


namespace gko {
namespace config {


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


// ResidualNorm and ImplicitResidualNorm have the same parameters
#define GKO_WRITE_RESIDUAL_NORM(_type)                                  \
    template <typename ValueType>                                       \
    pnode parameter_writer<stop::_type<ValueType>>::write(              \
        const typename stop::_type<ValueType>::parameters_type& params) \
    {                                                                   \
        auto map = make_config_map<ValueType>(#_type);                  \
        map["reduction_factor"] = write_value(params.reduction_factor); \
        map["baseline"] = write_value(params.baseline);                 \
        return pnode{map};                                              \
    }                                                                   \
    GKO_INSTANTIATE_WRITER_VALUE_TYPE(stop::_type)

GKO_WRITE_RESIDUAL_NORM(ResidualNorm);
GKO_WRITE_RESIDUAL_NORM(ImplicitResidualNorm);


}  // namespace config
}  // namespace gko

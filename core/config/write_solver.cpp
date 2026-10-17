// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/solver/cg.hpp>

#include "core/config/write_helper.hpp"


namespace gko {
namespace config {


// a solver whose parse reads only the common solver parameters
#define GKO_WRITE_COMMON_SOLVER(_type)                                    \
    template <typename ValueType>                                         \
    pnode parameter_writer<solver::_type<ValueType>>::write(              \
        const typename solver::_type<ValueType>::parameters_type& params) \
    {                                                                     \
        auto map = make_config_map<ValueType>("solver::" #_type);         \
        write_common_solver(map, params);                                 \
        return pnode{map};                                                \
    }                                                                     \
    GKO_INSTANTIATE_WRITER_VALUE_TYPE(solver::_type)


GKO_WRITE_COMMON_SOLVER(Cg);


}  // namespace config
}  // namespace gko

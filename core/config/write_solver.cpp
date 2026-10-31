// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/minres.hpp>
#include <ginkgo/core/solver/pipe_cg.hpp>
#include <ginkgo/core/solver/triangular.hpp>

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


// a triangular solver, whose parse reads only the common trisolver parameters
#define GKO_WRITE_TRISOLVER(_type)                                           \
    template <typename ValueType, typename IndexType>                        \
    pnode parameter_writer<solver::_type<ValueType, IndexType>>::write(      \
        const typename solver::_type<ValueType, IndexType>::parameters_type& \
            params)                                                          \
    {                                                                        \
        auto map = make_config_map<ValueType>("solver::" #_type);            \
        write_common_trisolver(map, params);                                 \
        return pnode{map};                                                   \
    }                                                                        \
    GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(solver::_type)


GKO_WRITE_COMMON_SOLVER(Cg);
GKO_WRITE_COMMON_SOLVER(Bicg);
GKO_WRITE_COMMON_SOLVER(Cgs);
GKO_WRITE_COMMON_SOLVER(Fcg);
GKO_WRITE_COMMON_SOLVER(PipeCg);
GKO_WRITE_COMMON_SOLVER(Minres);
GKO_WRITE_COMMON_SOLVER(Bicgstab);


template <typename ValueType>
pnode parameter_writer<solver::Gmres<ValueType>>::write(
    const typename solver::Gmres<ValueType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("solver::Gmres");
    write_common_solver(map, params);
    map["krylov_dim"] = write_value(params.krylov_dim);
    map["restart_ratio"] = write_value(params.restart_ratio);
    map["flexible"] = write_value(params.flexible);
    map["ortho_method"] = write_value(params.ortho_method);
    return pnode{map};
}

GKO_INSTANTIATE_WRITER_VALUE_TYPE(solver::Gmres);


// Ir does not read a preconditioner
template <typename ValueType>
pnode parameter_writer<solver::Ir<ValueType>>::write(
    const typename solver::Ir<ValueType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("solver::Ir");
    map["criteria"] = write_criteria(params.criteria);
    // an unset solver is the identity (see the Ir constructor)
    write_factory_with_identity_default(map, "solver", params.solver);
    write_object(map, "generated_solver", params.generated_solver);
    map["relaxation_factor"] = write_value(params.relaxation_factor);
    map["default_initial_guess"] = write_value(params.default_initial_guess);
    map["inner_initial_guess"] = write_value(params.inner_initial_guess);
    return pnode{map};
}

GKO_INSTANTIATE_WRITER_VALUE_TYPE(solver::Ir);


GKO_WRITE_TRISOLVER(LowerTrs);
GKO_WRITE_TRISOLVER(UpperTrs);


}  // namespace config
}  // namespace gko

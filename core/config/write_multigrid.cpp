// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>
#include <ginkgo/core/multigrid/uniform_coarsening.hpp>
#include <ginkgo/core/solver/multigrid.hpp>

#include "core/config/write_helper.hpp"


namespace gko {
namespace config {


// mirrors Multigrid::parse in core/solver/multigrid.cpp. Multigrid has no
// value type; its nested factories write their own. parse does not read the
// function-valued level_selector and solver_selector, so they are not written.
pnode parameter_writer<solver::Multigrid>::write(
    const solver::Multigrid::parameters_type& params)
{
    auto map = make_config_map("solver::Multigrid");
    map["criteria"] = write_criteria(params.criteria);
    // An empty list is left out. It is not the identity: Multigrid fails to
    // generate without an mg_level, uses one IR step with a scalar Jacobi
    // without a smoother, and a direct LU without a coarsest_solver (see
    // handle_list and generate_coarsest_solver).
    write_factory_list(map, "mg_level", params.mg_level);
    write_factory_list(map, "pre_smoother", params.pre_smoother);
    write_factory_list(map, "post_smoother", params.post_smoother);
    write_factory_list(map, "mid_smoother", params.mid_smoother);
    map["post_uses_pre"] = write_value(params.post_uses_pre);
    map["mid_case"] = write_value(params.mid_case);
    map["max_levels"] = write_value(params.max_levels);
    map["min_coarse_rows"] = write_value(params.min_coarse_rows);
    write_factory_list(map, "coarsest_solver", params.coarsest_solver);
    map["cycle"] = write_value(params.cycle);
    map["kcycle_base"] = write_value(params.kcycle_base);
    map["kcycle_rel_tol"] = write_value(params.kcycle_rel_tol);
    map["smoother_relax"] = write_value(params.smoother_relax);
    map["smoother_iters"] = write_value(params.smoother_iters);
    map["default_initial_guess"] = write_value(params.default_initial_guess);
    return pnode{map};
}


// mirrors Pgm::parse in core/multigrid/pgm.cpp
template <typename ValueType, typename IndexType>
pnode parameter_writer<multigrid::Pgm<ValueType, IndexType>>::write(
    const typename multigrid::Pgm<ValueType, IndexType>::parameters_type&
        params)
{
    auto map = make_config_map<ValueType>("multigrid::Pgm");
    map["max_iterations"] = write_value(params.max_iterations);
    map["max_unassigned_ratio"] = write_value(params.max_unassigned_ratio);
    map["deterministic"] = write_value(params.deterministic);
    map["skip_sorting"] = write_value(params.skip_sorting);
    return pnode{map};
}

#define GKO_DECLARE_PGM_WRITER(ValueType, IndexType) \
    struct parameter_writer<multigrid::Pgm<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM_WRITER);


// mirrors Pmis::parse in core/multigrid/pmis.cpp
template <typename ValueType, typename IndexType>
pnode parameter_writer<multigrid::Pmis<ValueType, IndexType>>::write(
    const typename multigrid::Pmis<ValueType, IndexType>::parameters_type&
        params)
{
    auto map = make_config_map<ValueType>("multigrid::Pmis");
    map["strength_threshold"] = write_value(params.strength_threshold);
    map["skip_sorting"] = write_value(params.skip_sorting);
    return pnode{map};
}

#define GKO_DECLARE_PMIS_WRITER(ValueType, IndexType) \
    struct parameter_writer<multigrid::Pmis<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS_WRITER);


// mirrors UniformCoarsening::parse in core/multigrid/uniform_coarsening.cpp
template <typename ValueType, typename IndexType>
pnode parameter_writer<multigrid::UniformCoarsening<ValueType, IndexType>>::
    write(const typename multigrid::UniformCoarsening<
          ValueType, IndexType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("multigrid::UniformCoarsening");
    map["coarse_skip"] = write_value(params.coarse_skip);
    map["aggregation"] = write_value(params.aggregation);
    map["skip_sorting"] = write_value(params.skip_sorting);
    return pnode{map};
}

#define GKO_DECLARE_UNIFORM_COARSENING_WRITER(ValueType, IndexType) \
    struct parameter_writer<multigrid::UniformCoarsening<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UNIFORM_COARSENING_WRITER);


}  // namespace config
}  // namespace gko

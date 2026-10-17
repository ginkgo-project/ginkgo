// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP
#define GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP


#include <string>
#include <utility>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/types.hpp"


/**
 * The kind of an operator and its number of nonzeros, as
 * {"type", "nonzeros"}. The type is one of csr, sparsity_csr (one nonzero per
 * stored index), row_gatherer (one gathered entry per row) or unknown. The
 * nonzeros are null for an unknown type.
 */
inline std::pair<std::string, json> describe_operator(const gko::LinOp* op)
{
    if (auto csr = dynamic_cast<const gko::matrix::Csr<etype, itype>*>(op)) {
        return {"csr", csr->get_num_stored_elements()};
    }
    if (auto sp =
            dynamic_cast<const gko::matrix::SparsityCsr<etype, itype>*>(op)) {
        return {"sparsity_csr", sp->get_num_nonzeros()};
    }
    if (auto rg = dynamic_cast<const gko::matrix::RowGatherer<itype>*>(op)) {
        return {"row_gatherer", rg->get_size()[0]};
    }
    return {"unknown", nullptr};
}


/**
 * Describes the levels of a multigrid solver, one object per level, with the
 * rows of the fine and coarse operators and the nonzeros of the coarse,
 * restriction and prolongation operators. A multigrid that did not coarsen
 * gives an empty array. Any other operator gives a null json.
 */
inline json describe_levels(const gko::LinOp* op)
{
    auto mg = dynamic_cast<const gko::solver::Multigrid*>(op);
    if (!mg) {
        return nullptr;
    }
    auto levels = json::array();
    for (const auto& level : mg->get_mg_level_list()) {
        json entry;
        entry["fine_rows"] = level->get_fine_op()->get_size()[0];
        entry["coarse_rows"] = level->get_coarse_op()->get_size()[0];
        entry["coarse_nonzeros"] =
            describe_operator(level->get_coarse_op().get()).second;
        auto restrict_op = describe_operator(level->get_restrict_op().get());
        auto prolong_op = describe_operator(level->get_prolong_op().get());
        entry["restrict_type"] = restrict_op.first;
        entry["restrict_nonzeros"] = restrict_op.second;
        entry["prolong_type"] = prolong_op.first;
        entry["prolong_nonzeros"] = prolong_op.second;
        levels.push_back(entry);
    }
    return levels;
}


#endif  // GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP

// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP
#define GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP


#include <memory>
#include <string>
#include <utility>
#include <vector>

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
 * Describes the smoother at index `level` of a list of generated smoothers of
 * a multigrid solver, as {type, solver type}: the demangled type of the
 * smoother, and for an IR smoother of the value type etype (e.g. Multigrid's
 * default smoother) the type of its inner solver. Each is null if there is
 * none, e.g. for a level without a smoother.
 */
inline std::pair<json, json> describe_smoother(
    const std::vector<std::shared_ptr<const gko::LinOp>>& smoothers,
    gko::size_type level)
{
    if (level >= smoothers.size() || !smoothers[level]) {
        return {nullptr, nullptr};
    }
    const auto& smoother = smoothers[level];
    json solver = nullptr;
    if (auto ir = dynamic_cast<const gko::solver::Ir<etype>*>(smoother.get());
        ir && ir->get_solver()) {
        solver = gko::name_demangling::get_dynamic_type(*ir->get_solver());
    }
    return {gko::name_demangling::get_dynamic_type(*smoother), solver};
}


/**
 * Describes the levels of a multigrid solver, one object per level, with the
 * rows of the fine and coarse operators, the nonzeros of the coarse,
 * restriction and prolongation operators, and the types of the generated pre-,
 * mid- and post-smoothers (see describe_smoother). A multigrid that did not
 * coarsen gives an empty array. Any other operator gives a null json.
 */
inline json describe_levels(const gko::LinOp* op)
{
    auto mg = dynamic_cast<const gko::solver::Multigrid*>(op);
    if (!mg) {
        return nullptr;
    }
    const auto smoothers = {
        std::make_pair("pre_smoother", mg->get_pre_smoother_list()),
        std::make_pair("mid_smoother", mg->get_mid_smoother_list()),
        std::make_pair("post_smoother", mg->get_post_smoother_list())};
    auto levels = json::array();
    const auto& level_list = mg->get_mg_level_list();
    for (gko::size_type i = 0; i < level_list.size(); ++i) {
        const auto& level = level_list[i];
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
        for (const auto& [name, list] : smoothers) {
            auto [type, solver] = describe_smoother(list, i);
            entry[name] = type;
            entry[std::string{name} + "_solver"] = solver;
        }
        levels.push_back(entry);
    }
    return levels;
}


#endif  // GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP

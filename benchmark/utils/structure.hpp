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
 * The type and nonzeros of a csr, sparsity_csr or row_gatherer operator, or
 * "unknown" and null.
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
 * The type of a generated smoother of a multigrid and, for an IR smoother,
 * of its inner solver; null if there is none.
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
 * The rows, nonzeros and smoothers of each multigrid level, null for other
 * operators.
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

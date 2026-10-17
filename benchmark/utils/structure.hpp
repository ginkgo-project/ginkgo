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
 * The rows and nonzeros of each multigrid level, null for other operators.
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

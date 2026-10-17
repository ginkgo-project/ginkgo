// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP
#define GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP


#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/types.hpp"


/**
 * The number of stored elements of op if it is a CSR matrix, null otherwise.
 */
inline json csr_nonzeros(const gko::LinOp* op)
{
    if (auto csr = dynamic_cast<const gko::matrix::Csr<etype, itype>*>(op)) {
        return csr->get_num_stored_elements();
    }
    return nullptr;
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
        entry["coarse_nonzeros"] = csr_nonzeros(level->get_coarse_op().get());
        entry["restrict_nonzeros"] =
            csr_nonzeros(level->get_restrict_op().get());
        entry["prolong_nonzeros"] = csr_nonzeros(level->get_prolong_op().get());
        levels.push_back(entry);
    }
    return levels;
}


#endif  // GINKGO_BENCHMARK_UTILS_STRUCTURE_HPP

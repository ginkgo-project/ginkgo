// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/pmis.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/distributed/index_map_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/pmis_kernels.hpp"


namespace gko {
namespace multigrid {
namespace pmis {
namespace {


GKO_REGISTER_OPERATION(compute_strong_dep_row, pmis::compute_strong_dep_row);
GKO_REGISTER_OPERATION(compute_strong_dep, pmis::compute_strong_dep);
GKO_REGISTER_OPERATION(initialize_weight_and_status,
                       pmis::initialize_weight_and_status);
GKO_REGISTER_OPERATION(classify, pmis::classify);
GKO_REGISTER_OPERATION(count, pmis::count);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);


}  // anonymous namespace
}  // namespace pmis


template <typename ValueType, typename IndexType>
typename Pmis<ValueType, IndexType>::parameters_type
Pmis<ValueType, IndexType>::parse(const config::pnode& config,
                                  const config::registry& context,
                                  const config::type_descriptor& td_for_child)
{
    auto params = Pmis<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("strength_threshold")) {
        params.with_strength_threshold(
            config::get_value<remove_complex<ValueType>>(obj));
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }

    return params;
}


template <typename ValueType, typename IndexType>
void Pmis<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    // Only support csr matrix currently.
    auto pmis_op = std::dynamic_pointer_cast<const csr_type>(system_matrix_);
    // If system matrix is not csr or need sorting, generate the csr.
    if (!parameters_.skip_sorting || !pmis_op) {
        pmis_op = convert_to_with_sorting<csr_type>(exec, system_matrix_,
                                                    parameters_.skip_sorting);
        // keep the same precision data in fine_op
        this->set_fine_op(pmis_op);
    }

    array<IndexType> sparsity_rows(exec, pmis_op->get_size()[0] + 1);
    // the number of #S_i into sparsity_row i
    exec->run(pmis::make_compute_strong_dep_row(
        pmis_op.get(), this->get_parameters().strength_threshold,
        sparsity_rows.get_data()));
    // build offset
    exec->run(pmis::make_prefix_sum_nonnegative(sparsity_rows.get_data(),
                                                sparsity_rows.get_size()));
    auto nnz = exec->copy_val_to_host(sparsity_rows.get_const_data() +
                                      pmis_op->get_size()[0]);
    array<IndexType> sparsity_cols(exec, nnz);
    auto strong_dep = matrix::SparsityCsr<ValueType, IndexType>::create(
        exec, pmis_op->get_size(), std::move(sparsity_cols),
        std::move(sparsity_rows));
    // fill column index into sparsity csr
    exec->run(pmis::make_compute_strong_dep(
        pmis_op.get(), this->get_parameters().strength_threshold,
        strong_dep.get()));
    // weight[i] = #S^T + rand(0, 1)
    // status 0: not assigned, 1: fine group 2: coarse group
    // status[i] = 1 if #S^T_i = 0 or 0
    exec->run(pmis::make_initialize_weight_and_status(
        strong_dep.get(), weight_.get_data(), status_.get_data()));
    size_type num_not_assigned = 0;
    // count #{status == 0}
    exec->run(pmis::make_count(status_, &num_not_assigned));
    while (num_not_assigned != 0) {
        exec->run(pmis::make_classify(weight_.get_const_data(),
                                      strong_dep.get(), status_.get_data()));
        size_type new_num = 0;
        exec->run(pmis::make_count(status_, &new_num));
        if (new_num == num_not_assigned) {
            // no progess -> throw error (maybe unneccessary)
            throw std::runtime_error("no progress in Pmis");
        }
        num_not_assigned = new_num;
    }
    // finish classify points to fine and coarse group.
}


#define GKO_DECLARE_PMIS(_vtype, _itype) class Pmis<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS);


}  // namespace multigrid
}  // namespace gko

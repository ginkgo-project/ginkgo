// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
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

#include "core/base/array_access.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/precision_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/distributed/index_map_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/pmis_kernels.hpp"


namespace gko {
namespace multigrid {
namespace pmis {
namespace {


GKO_REGISTER_OPERATION(compute_row_maxabs, pmis::compute_row_maxabs);
GKO_REGISTER_OPERATION(compute_strong_dep_row, pmis::compute_strong_dep_row);
GKO_REGISTER_OPERATION(compute_strong_dep, pmis::compute_strong_dep);
GKO_REGISTER_OPERATION(initialize_weight_and_status,
                       pmis::initialize_weight_and_status);
GKO_REGISTER_OPERATION(classify, pmis::classify);
GKO_REGISTER_OPERATION(count, pmis::count);
GKO_REGISTER_OPERATION(direct_interpolation_row_count,
                       pmis::direct_interpolation_row_count);
GKO_REGISTER_OPERATION(direct_interpolation_fill,
                       pmis::direct_interpolation_fill);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(convert_precision, components::convert_precision);


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
void Pmis<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    this->get_composition()->apply(b, x);
}


template <typename ValueType, typename IndexType>
void Pmis<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                            const LinOp* beta, LinOp* x) const
{
    this->get_composition()->apply(alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
Pmis<ValueType, IndexType>::Pmis(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec))
{}


template <typename ValueType, typename IndexType>
Pmis<ValueType, IndexType>::Pmis(const Factory* factory,
                                 std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), system_matrix->get_size()),
      EnableMultigridLevel<ValueType>(system_matrix),
      parameters_{factory->get_parameters()},
      system_matrix_{system_matrix}
{
    GKO_ASSERT(parameters_.strength_threshold <= 1.0);
    GKO_ASSERT(parameters_.strength_threshold >= 0.0);
    if (system_matrix_->get_size()[0] != 0) {
        // generate on the existed matrix
        this->generate();
    }
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
    array<remove_complex<ValueType>> row_maxabs(exec, pmis_op->get_size()[0]);
    // weight = the number of strong dependence + rand[0, 1]
    gko::array<remove_complex<ValueType>> weight_(exec, pmis_op->get_size()[0]);
    exec->run(
        pmis::make_compute_row_maxabs(pmis_op.get(), row_maxabs.get_data()));
    // the number of nonzero in strong_dep of node i (#S_i) into sparsity_row i
    exec->run(pmis::make_compute_strong_dep_row(
        pmis_op.get(), row_maxabs.get_const_data(),
        this->get_parameters().strength_threshold, sparsity_rows.get_data()));
    // build offset
    exec->run(pmis::make_prefix_sum_nonnegative(sparsity_rows.get_data(),
                                                sparsity_rows.get_size()));
    auto nnz = get_element(sparsity_rows, pmis_op->get_size()[0]);
    array<IndexType> sparsity_cols(exec, nnz);
    auto strong_dep = matrix::SparsityCsr<ValueType, IndexType>::create(
        exec, pmis_op->get_size(), std::move(sparsity_cols),
        std::move(sparsity_rows));
    // fill column index into sparsity csr
    exec->run(pmis::make_compute_strong_dep(
        pmis_op.get(), row_maxabs.get_const_data(),
        this->get_parameters().strength_threshold, strong_dep.get()));
    auto transpose_strong_dep =
        as<matrix::SparsityCsr<ValueType, IndexType>>(strong_dep->transpose());
    // weight[i] = #S^T + rand(0, 1)
    // status -1: not assigned, 0: fine group 1: coarse group
    // status[i] = 0 if #S^T_i = 0 or -1
    gko::array<int> status(exec, this->get_size()[0]);
    gko::array<int> new_status(exec, this->get_size()[0]);
    auto status_ptr = status.get_data();
    auto new_status_ptr = new_status.get_data();
    exec->run(pmis::make_initialize_weight_and_status(
        transpose_strong_dep.get(), weight_.get_data(), status_ptr));
    size_type num_not_assigned = 0;
    // count #{status == 0}
    exec->run(
        pmis::make_count(this->get_size()[0], status_ptr, &num_not_assigned));
    while (num_not_assigned != 0) {
        exec->run(pmis::make_classify(weight_.get_const_data(), pmis_op.get(),
                                      transpose_strong_dep.get(), status_ptr,
                                      new_status_ptr));
        size_type new_num = 0;
        exec->run(
            pmis::make_count(this->get_size()[0], new_status_ptr, &new_num));
        GKO_THROW_IF_INVALID(new_num != num_not_assigned,
                             "no progress in Pmis");
        num_not_assigned = new_num;
        std::swap(new_status_ptr, status_ptr);
    }
    // finish classify points to fine and coarse group.
    array<IndexType> prolong_row_ptrs(exec, pmis_op->get_size()[0] + 1);
    exec->run(pmis::make_direct_interpolation_row_count(
        strong_dep.get(), status_ptr, prolong_row_ptrs.get_data()));

    // coarse_map[i] gives the coarse index from i if i will appear in coarse
    // grid. if i is not in coarse grid, coarse_map[i] has no meaning; In
    // theory, we can reuse status_ptr as coarse_map. We iterate the classify
    // process on status_ptr, so keeping that it in int not IndexType might give
    // some performance benefit.
    array<IndexType> coarse_map(exec, pmis_op->get_size()[0] + 1);
    exec->run(pmis::make_convert_precision(pmis_op->get_size()[0], status_ptr,
                                           coarse_map.get_data()));
    exec->run(pmis::make_prefix_sum_nonnegative(coarse_map.get_data(),
                                                this->get_size()[0] + 1));
    auto num_coarse =
        static_cast<size_type>(get_element(coarse_map, this->get_size()[0]));

    // the following implements direct interpolation, c is coarse_map, which map
    // the fine grid index k to coarse grid index c[k] if k will appear in
    // coarse grid. Construct interpolation W which contain value w_{i, c[i]} if
    // i will appear in coarse grid or w_{i, c[k]} if S_ik exists and k will
    // appear in coarse grid.

    exec->run(pmis::make_prefix_sum_nonnegative(prolong_row_ptrs.get_data(),
                                                this->get_size()[0] + 1));
    IndexType prolong_nnz = get_element(prolong_row_ptrs, this->get_size()[0]);
    array<IndexType> prolong_col_idxs(exec, prolong_nnz);
    array<ValueType> prolong_values(exec, prolong_nnz);

    // alpha_i = sum(a_ij which a_ij < 0) / sum(a_ik which s_ik exist, a_ik < 0,
    // k is coarse point)
    // beta_i = sum(a_ij which a_ij > 0) / sum(a_ik which
    // s_ik exist, a_ik > 0, k is coarse point)
    // If there is no entry for alpha_i or beta_i, we consider it is empty.
    // If the following formula uses empty alpha_i or beta_i, the weight will
    // not have that entry. finish weight construction w_{i, c[k]} = {-alpha_i
    // if a_ik is negative or -beta_i if it is positive} * a_ik/a_ii if i is
    // fine point. If i is coarse point w_{i, c[i]} = 1
    exec->run(pmis::make_direct_interpolation_fill(
        pmis_op.get(), row_maxabs.get_const_data(),
        this->get_parameters().strength_threshold, coarse_map.get_const_data(),
        prolong_row_ptrs.get_const_data(), prolong_col_idxs.get_data(),
        prolong_values.get_data()));
    auto prolongation = share(matrix::Csr<ValueType, IndexType>::create(
        exec, dim<2>{pmis_op->get_size()[0], num_coarse},
        std::move(prolong_values), std::move(prolong_col_idxs),
        std::move(prolong_row_ptrs)));
    auto restriction = share(prolongation->transpose());
    auto internal = matrix::Csr<ValueType, IndexType>::create(
        exec, prolongation->get_size());
    auto coarse = share(matrix::Csr<ValueType, IndexType>::create(
        exec, dim<2>{num_coarse, num_coarse}));
    pmis_op->apply(prolongation, internal);
    restriction->apply(internal, coarse);
    this->set_multigrid_level(prolongation, coarse, restriction);
}


#define GKO_DECLARE_PMIS(_vtype, _itype) class Pmis<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS);


}  // namespace multigrid
}  // namespace gko

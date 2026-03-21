// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/hmis.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>

#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/hmis_kernels.hpp"


namespace gko {
namespace multigrid {
namespace hmis {
namespace {


GKO_REGISTER_OPERATION(compute_strength, hmis::compute_strength);
GKO_REGISTER_OPERATION(ruge_stueben_first_pass, hmis::ruge_stueben_first_pass);
GKO_REGISTER_OPERATION(pmis_coarsen, hmis::pmis_coarsen);
GKO_REGISTER_OPERATION(build_cpoint_map, hmis::build_cpoint_map);
GKO_REGISTER_OPERATION(build_prolongation, hmis::build_prolongation);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);


}  // anonymous namespace
}  // namespace hmis


template <typename ValueType, typename IndexType>
typename Hmis<ValueType, IndexType>::parameters_type
Hmis<ValueType, IndexType>::parse(const config::pnode& config,
                                  const config::registry& context,
                                  const config::type_descriptor& td_for_child)
{
    auto params = Hmis<ValueType, IndexType>::build();
    if (auto& obj = config.get("strength_threshold")) {
        params.with_strength_threshold(gko::config::get_value<double>(obj));
    }
    if (auto& obj = config.get("max_iterations")) {
        params.with_max_iterations(gko::config::get_value<unsigned>(obj));
    }
    if (auto& obj = config.get("skip_sorting")) {
        params.with_skip_sorting(gko::config::get_value<bool>(obj));
    }
    return params;
}


template <typename ValueType, typename IndexType>
std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
           std::shared_ptr<LinOp>>
Hmis<ValueType, IndexType>::generate_local(
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> local_matrix)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    auto exec = this->get_executor();
    const auto num_rows = static_cast<IndexType>(local_matrix->get_size()[0]);
    const auto fine_nnz = local_matrix->get_num_stored_elements();

    // Phase 1: Compute strength of connection
    // First pass: compute row_ptrs for the strength matrix
    array<IndexType> strength_row_ptrs_array(exec, num_rows + 1);
    array<IndexType> strength_col_idxs_array(exec);
    exec->run(hmis::make_compute_strength(
        local_matrix.get(),
        static_cast<real_type>(parameters_.strength_threshold),
        strength_row_ptrs_array.get_data(), strength_col_idxs_array));
    // Build strength CSR (sparsity-only, values are 1)
    const auto strength_nnz = strength_col_idxs_array.get_size();
    array<ValueType> strength_vals(exec, strength_nnz);
    exec->run(hmis::make_fill_array(strength_vals.get_data(), strength_nnz,
                                    one<ValueType>()));
    auto strength = csr_type::create(
        exec,
        gko::dim<2>{static_cast<size_type>(num_rows),
                    static_cast<size_type>(num_rows)},
        std::move(strength_vals), std::move(strength_col_idxs_array),
        std::move(strength_row_ptrs_array));

    // Compute transpose of strength
    auto strength_transpose = gko::as<csr_type>(share(strength->transpose()));

    // Phase 2: C/F splitting
    cf_marker_.resize_and_reset(num_rows);
    // Ruge-Stueben first pass
    exec->run(hmis::make_ruge_stueben_first_pass(
        strength.get(), strength_transpose.get(), cf_marker_));
    // PMIS finalization for any remaining undecided points
    exec->run(hmis::make_pmis_coarsen(strength.get(), strength_transpose.get(),
                                      parameters_.max_iterations, cf_marker_));

    // Phase 3: Build C-point map
    IndexType num_c_points = 0;
    array<IndexType> c_point_map(exec, num_rows);
    exec->run(
        hmis::make_build_cpoint_map(cf_marker_, &num_c_points, c_point_map));

    // Phase 4: Build prolongation via classical interpolation
    const auto coarse_dim = static_cast<size_type>(num_c_points);
    const auto fine_dim = static_cast<size_type>(num_rows);
    array<IndexType> prolong_row_ptrs_array(exec, num_rows + 1);
    array<IndexType> prolong_col_idxs_array(exec);
    array<ValueType> prolong_vals_array(exec);
    exec->run(hmis::make_build_prolongation(
        local_matrix.get(), strength.get(), cf_marker_, num_c_points,
        c_point_map, prolong_row_ptrs_array.get_data(), prolong_col_idxs_array,
        prolong_vals_array));

    auto prolong_op = share(csr_type::create(
        exec, gko::dim<2>{fine_dim, coarse_dim}, std::move(prolong_vals_array),
        std::move(prolong_col_idxs_array), std::move(prolong_row_ptrs_array)));

    // Phase 5: Restriction = P^T
    auto restrict_op = gko::as<csr_type>(share(prolong_op->transpose()));

    // Phase 6: Coarse matrix = R * A * P (Galerkin triple product)
    auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim});
    local_matrix->apply(prolong_op, tmp);
    auto coarse_matrix =
        share(csr_type::create(exec, gko::dim<2>{coarse_dim, coarse_dim}));
    restrict_op->apply(tmp, coarse_matrix);

    return std::make_tuple(std::move(prolong_op), std::move(coarse_matrix),
                           std::move(restrict_op));
}


template <typename ValueType, typename IndexType>
void Hmis<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    // Only support csr matrix currently.
    auto hmis_op = std::dynamic_pointer_cast<const csr_type>(system_matrix_);
    // If system matrix is not csr or need sorting, generate the csr.
    if (!parameters_.skip_sorting || !hmis_op) {
        hmis_op = convert_to_with_sorting<csr_type>(exec, system_matrix_,
                                                    parameters_.skip_sorting);
        this->set_fine_op(hmis_op);
    }
    auto result = this->generate_local(hmis_op);
    this->set_multigrid_level(std::get<0>(result), std::get<1>(result),
                              std::get<2>(result));
}


#define GKO_DECLARE_HMIS(_vtype, _itype) class Hmis<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HMIS);


}  // namespace multigrid
}  // namespace gko

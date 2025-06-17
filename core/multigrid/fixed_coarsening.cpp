// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/fixed_coarsening.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/fixed_coarsening_kernels.hpp"
// #include "core/multigrid/pgm_kernels.hpp"


namespace gko {
namespace multigrid {
namespace fixed_coarsening {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);
GKO_REGISTER_OPERATION(build_row_ptrs, fixed_coarsening::build_row_ptrs);
GKO_REGISTER_OPERATION(renumber, fixed_coarsening::renumber);
GKO_REGISTER_OPERATION(map_to_coarse, fixed_coarsening::map_to_coarse);


}  // anonymous namespace
}  // namespace fixed_coarsening


// selected_rows is the sorted index will be used
// selected_cols_map gives the new col index from the old col index (invalid)
template <typename ValueType, typename IndexType>
std::shared_ptr<matrix::Csr<ValueType, IndexType>> build_coarse_matrix(
    const matrix::Csr<ValueType, IndexType>* origin, const size_type num_cols,
    const array<IndexType>& selected_rows,
    const array<IndexType>& selected_cols_map)
{
    auto exec = origin->get_executor();
    auto coarse = matrix::Csr<ValueType, IndexType>::create(
        exec, dim<2>{selected_rows.get_size(), num_cols});
    exec->run(fixed_coarsening::make_build_row_ptrs(
        origin->get_size()[0], origin->get_const_row_ptrs(),
        origin->get_const_col_idxs(), selected_rows, selected_cols_map,
        coarse->get_size()[0], coarse->get_row_ptrs()));
    auto coarse_nnz = static_cast<size_type>(
        exec->copy_val_to_host(coarse->get_row_ptrs() + coarse->get_size()[0]));
    array<ValueType> new_value_array{exec, coarse_nnz};
    array<IndexType> new_col_idx_array{exec, coarse_nnz};
    exec->run(fixed_coarsening::make_map_to_coarse(
        origin->get_size()[0], origin->get_const_row_ptrs(),
        origin->get_const_col_idxs(), origin->get_const_values(), selected_rows,
        selected_cols_map, coarse->get_size()[0], coarse->get_const_row_ptrs(),
        new_col_idx_array.get_data(), new_value_array.get_data()));
    matrix::CsrBuilder<ValueType, IndexType> mtx_builder{coarse};
    mtx_builder.get_value_array() = std::move(new_value_array);
    mtx_builder.get_col_idx_array() = std::move(new_col_idx_array);
    return coarse;
}


template <typename ValueType, typename IndexType>
void FixedCoarsening<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    auto exec = this->get_executor();
    const auto num_rows = this->system_matrix_->get_size()[0];

    // Only support csr matrix currently.
    const csr_type* fixed_coarsening_op =
        dynamic_cast<const csr_type*>(system_matrix_.get());
    std::shared_ptr<const csr_type> fixed_coarsening_op_shared_ptr{};
    // If system matrix is not csr or need sorting, generate the csr.
    if (!parameters_.skip_sorting || !fixed_coarsening_op) {
        fixed_coarsening_op_shared_ptr = convert_to_with_sorting<csr_type>(
            exec, system_matrix_, parameters_.skip_sorting);
        fixed_coarsening_op = fixed_coarsening_op_shared_ptr.get();
        // keep the same precision data in fine_op
        this->set_fine_op(fixed_coarsening_op_shared_ptr);
    }

    GKO_ASSERT(parameters_.coarse_rows.get_data() != nullptr);
    GKO_ASSERT(parameters_.coarse_rows.get_size() > 0);
    size_type coarse_dim = parameters_.coarse_rows.get_size();

    auto fine_dim = system_matrix_->get_size()[0];
    auto restrict_op = share(
        csr_type::create(exec, gko::dim<2>{coarse_dim, fine_dim}, coarse_dim,
                         fixed_coarsening_op->get_strategy()));
    exec->copy_from(parameters_.coarse_rows.get_executor(), coarse_dim,
                    parameters_.coarse_rows.get_const_data(),
                    restrict_op->get_col_idxs());
    exec->run(fixed_coarsening::make_fill_array(restrict_op->get_values(),
                                                coarse_dim, one<ValueType>()));
    exec->run(fixed_coarsening::make_fill_seq_array(restrict_op->get_row_ptrs(),
                                                    coarse_dim + 1));

    auto prolong_op = gko::as<csr_type>(share(restrict_op->transpose()));

    // generate the map from coarse_row. map[i] -> new index
    // it may gives some additional work for local case, but it gives the
    // neccessary information for distributed case
    array<IndexType> coarse_map(exec, fine_dim);
    coarse_map.fill(invalid_index<IndexType>());
    exec->run(
        fixed_coarsening::make_renumber(parameters_.coarse_rows, &coarse_map));
    // TODO: Can be done with submatrix index_set.
    auto coarse_matrix = build_coarse_matrix(
        fixed_coarsening_op, coarse_dim, parameters_.coarse_rows, coarse_map);
    coarse_matrix->set_strategy(fixed_coarsening_op->get_strategy());

    this->set_multigrid_level(prolong_op, coarse_matrix, restrict_op);
}


#define GKO_DECLARE_FIXED_COARSENING(_vtype, _itype) \
    class FixedCoarsening<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FIXED_COARSENING);


}  // namespace multigrid
}  // namespace gko

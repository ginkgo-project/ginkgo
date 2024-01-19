// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/multigrid/uniform_coarsening.hpp>


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
#include "core/multigrid/uniform_coarsening_kernels.hpp"


namespace gko {
namespace multigrid {
namespace uniform_coarsening {
namespace {


GKO_REGISTER_OPERATION(fill_restrict_op, uniform_coarsening::fill_restrict_op);
GKO_REGISTER_OPERATION(fill_incremental_indices,
                       uniform_coarsening::fill_incremental_indices);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // anonymous namespace
}  // namespace uniform_coarsening


template <typename ValueType, typename IndexType>
void UniformCoarsening<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using sparsity_type = matrix::SparsityCsr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    using weight_csr_type = remove_complex<csr_type>;
    auto exec = this->get_executor();
    const auto num_rows = this->system_matrix_->get_size()[0];

    // Only support csr matrix currently.
    const csr_type* uniform_coarsening_op =
        dynamic_cast<const csr_type*>(system_matrix_.get());
    std::shared_ptr<const csr_type> uniform_coarsening_op_shared_ptr{};
    // If system matrix is not csr or need sorting, generate the csr.
    if (!parameters_.skip_sorting || !uniform_coarsening_op) {
        uniform_coarsening_op_shared_ptr = convert_to_with_sorting<csr_type>(
            exec, system_matrix_, parameters_.skip_sorting);
        uniform_coarsening_op = uniform_coarsening_op_shared_ptr.get();
        // keep the same precision data in fine_op
        this->set_fine_op(uniform_coarsening_op_shared_ptr);
    }
    // Use -1 as sentinel value
    coarse_rows_ = array<IndexType>(exec, num_rows);
    coarse_rows_.fill(-one<IndexType>());

    // Fill with incremental local indices.
    exec->run(uniform_coarsening::make_fill_incremental_indices(
        parameters_.coarse_skip, &coarse_rows_));

    gko::dim<2>::dimension_type coarse_dim =
        (coarse_rows_.get_size() + parameters_.coarse_skip - 1) /
        parameters_.coarse_skip;
    auto fine_dim = system_matrix_->get_size()[0];
    auto restrict_op = share(
        csr_type::create(exec, gko::dim<2>{coarse_dim, fine_dim}, coarse_dim,
                         uniform_coarsening_op->get_strategy()));
    exec->run(uniform_coarsening::make_fill_restrict_op(&coarse_rows_,
                                                        restrict_op.get()));
    exec->run(uniform_coarsening::make_fill_array(
        restrict_op->get_values(), coarse_dim, one<ValueType>()));
    exec->run(uniform_coarsening::make_fill_seq_array(
        restrict_op->get_row_ptrs(), coarse_dim + 1));

    auto prolong_op = gko::as<csr_type>(share(restrict_op->transpose()));

    // TODO: Can be done with submatrix index_set.
    auto coarse_matrix =
        share(csr_type::create(exec, gko::dim<2>{coarse_dim, coarse_dim}));
    coarse_matrix->set_strategy(uniform_coarsening_op->get_strategy());
    auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim});
    tmp->set_strategy(uniform_coarsening_op->get_strategy());
    uniform_coarsening_op->apply(prolong_op, tmp);
    restrict_op->apply(tmp, coarse_matrix);

    this->set_multigrid_level(prolong_op, coarse_matrix, restrict_op);
}


#define GKO_DECLARE_UNIFORM_COARSENING(_vtype, _itype) \
    class UniformCoarsening<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UNIFORM_COARSENING);


}  // namespace multigrid
}  // namespace gko

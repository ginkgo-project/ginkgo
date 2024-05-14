// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/multigrid/fixed_coarsening.hpp>


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


namespace gko {
namespace multigrid {
namespace fixed_coarsening {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // anonymous namespace
}  // namespace fixed_coarsening


template <typename ValueType, typename IndexType>
void FixedCoarsening<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using sparsity_type = matrix::SparsityCsr<ValueType, IndexType>;
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

    // TODO: Can be done with submatrix index_set.
    auto coarse_matrix =
        share(csr_type::create(exec, gko::dim<2>{coarse_dim, coarse_dim}));
    coarse_matrix->set_strategy(fixed_coarsening_op->get_strategy());
    auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim});
    tmp->set_strategy(fixed_coarsening_op->get_strategy());
    fixed_coarsening_op->apply(prolong_op, tmp);
    restrict_op->apply(tmp, coarse_matrix);

    this->set_multigrid_level(prolong_op, coarse_matrix, restrict_op);
}


#define GKO_DECLARE_FIXED_COARSENING(_vtype, _itype) \
    class FixedCoarsening<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FIXED_COARSENING);


}  // namespace multigrid
}  // namespace gko

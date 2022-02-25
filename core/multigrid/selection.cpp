/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/multigrid/selection.hpp>


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
#include "core/multigrid/selection_kernels.hpp"


namespace gko {
namespace multigrid {
namespace selection {
namespace {


GKO_REGISTER_OPERATION(fill_restrict_op, selection::fill_restrict_op);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // anonymous namespace
}  // namespace selection


template <typename ValueType, typename IndexType>
void Selection<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using sparsity_type = matrix::SparsityCsr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    using weight_csr_type = remove_complex<csr_type>;
    auto exec = this->get_executor();
    const auto num_rows = this->system_matrix_->get_size()[0];

    // Only support csr matrix currently.
    const csr_type* selection_op =
        dynamic_cast<const csr_type*>(system_matrix_.get());
    std::shared_ptr<const csr_type> selection_op_shared_ptr{};
    // If system matrix is not csr or need sorting, generate the csr.
    if (!parameters_.skip_sorting || !selection_op) {
        selection_op_shared_ptr = convert_to_with_sorting<csr_type>(
            exec, system_matrix_, parameters_.skip_sorting);
        selection_op = selection_op_shared_ptr.get();
        // keep the same precision data in fine_op
        this->set_fine_op(selection_op_shared_ptr);
    }
    // Use -1 as sentinel value
    auto diag_select = std::vector<IndexType>(num_rows, -1);

    // Fill with incremental local indices.
    size_type num_coarse_rows = 0;
    for (auto i = 0; i < num_rows; i += parameters_.num_jumps) {
        diag_select[i] = num_coarse_rows++;
    }
    coarse_rows_ = Array<IndexType>(exec, diag_select.data(),
                                    diag_select.data() + num_rows);

    gko::dim<2>::dimension_type coarse_dim = num_coarse_rows;
    auto fine_dim = system_matrix_->get_size()[0];
    // prolong_row_gather is the lightway implementation for prolongation
    // TODO: However, we still create the csr to process coarse/restrict matrix
    // generation. It may be changed when we have the direct triple product from
    // agg index.
    auto restrict_op =
        share(csr_type::create(exec, gko::dim<2>{coarse_dim, fine_dim},
                               coarse_dim, selection_op->get_strategy()));
    exec->run(
        selection::make_fill_restrict_op(&coarse_rows_, restrict_op.get()));
    exec->run(selection::make_fill_array(restrict_op->get_values(), coarse_dim,
                                         one<ValueType>()));
    exec->run(selection::make_fill_seq_array(restrict_op->get_row_ptrs(),
                                             coarse_dim + 1));

    auto prolong_op = gko::as<csr_type>(share(restrict_op->transpose()));

    // TODO: Can be done with submatrix index_set.
    auto coarse_matrix =
        share(csr_type::create(exec, gko::dim<2>{coarse_dim, coarse_dim}));
    coarse_matrix->set_strategy(selection_op->get_strategy());
    auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim});
    tmp->set_strategy(selection_op->get_strategy());
    selection_op->apply(prolong_op.get(), tmp.get());
    restrict_op->apply(tmp.get(), coarse_matrix.get());

    this->set_multigrid_level(prolong_op, coarse_matrix, restrict_op);
}


#define GKO_DECLARE_SELECTION(_vtype, _itype) class Selection<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELECTION);


}  // namespace multigrid
}  // namespace gko

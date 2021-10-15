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

#include <ginkgo/core/multigrid/amgx_pgm.hpp>


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
#include "core/multigrid/amgx_pgm_kernels.hpp"


namespace gko {
namespace multigrid {
namespace amgx_pgm {
namespace {


GKO_REGISTER_OPERATION(match_edge, amgx_pgm::match_edge);
GKO_REGISTER_OPERATION(count_unagg, amgx_pgm::count_unagg);
GKO_REGISTER_OPERATION(renumber, amgx_pgm::renumber);
GKO_REGISTER_OPERATION(find_strongest_neighbor,
                       amgx_pgm::find_strongest_neighbor);
GKO_REGISTER_OPERATION(assign_to_exist_agg, amgx_pgm::assign_to_exist_agg);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // anonymous namespace
}  // namespace amgx_pgm


template <typename ValueType, typename IndexType>
void AmgxPgm<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    using weight_csr_type = remove_complex<csr_type>;
    auto exec = this->get_executor();
    const auto num_rows = this->system_matrix_->get_size()[0];
    Array<IndexType> strongest_neighbor(this->get_executor(), num_rows);
    Array<IndexType> intermediate_agg(this->get_executor(),
                                      parameters_.deterministic * num_rows);
    // Only support csr matrix currently.
    const csr_type* amgxpgm_op =
        dynamic_cast<const csr_type*>(system_matrix_.get());
    std::shared_ptr<const csr_type> amgxpgm_op_shared_ptr{};
    // If system matrix is not csr or need sorting, generate the csr.
    if (!parameters_.skip_sorting || !amgxpgm_op) {
        amgxpgm_op_shared_ptr = convert_to_with_sorting<csr_type>(
            exec, system_matrix_, parameters_.skip_sorting);
        amgxpgm_op = amgxpgm_op_shared_ptr.get();
        // keep the same precision data in fine_op
        this->set_fine_op(amgxpgm_op_shared_ptr);
    }
    // Initial agg = -1
    exec->run(amgx_pgm::make_fill_array(agg_.get_data(), agg_.get_num_elems(),
                                        -one<IndexType>()));
    IndexType num_unagg = num_rows;
    IndexType num_unagg_prev = num_rows;
    // TODO: if mtx is a hermitian matrix, weight_mtx = abs(mtx)
    // compute weight_mtx = (abs(mtx) + abs(mtx'))/2;
    auto abs_mtx = amgxpgm_op->compute_absolute();
    // abs_mtx is already real valuetype, so transpose is enough
    auto weight_mtx = gko::as<weight_csr_type>(abs_mtx->transpose());
    auto half_scalar = initialize<matrix::Dense<real_type>>({0.5}, exec);
    auto identity = matrix::Identity<real_type>::create(exec, num_rows);
    // W = (abs_mtx + transpose(abs_mtx))/2
    abs_mtx->apply(lend(half_scalar), lend(identity), lend(half_scalar),
                   lend(weight_mtx));
    // Extract the diagonal value of matrix
    auto diag = weight_mtx->extract_diagonal();
    for (int i = 0; i < parameters_.max_iterations; i++) {
        // Find the strongest neighbor of each row
        exec->run(amgx_pgm::make_find_strongest_neighbor(
            weight_mtx.get(), diag.get(), agg_, strongest_neighbor));
        // Match edges
        exec->run(amgx_pgm::make_match_edge(strongest_neighbor, agg_));
        // Get the num_unagg
        exec->run(amgx_pgm::make_count_unagg(agg_, &num_unagg));
        // no new match, all match, or the ratio of num_unagg/num is lower
        // than parameter.max_unassigned_ratio
        if (num_unagg == 0 || num_unagg == num_unagg_prev ||
            num_unagg < parameters_.max_unassigned_ratio * num_rows) {
            break;
        }
        num_unagg_prev = num_unagg;
    }
    // Handle the left unassign points
    if (num_unagg != 0 && parameters_.deterministic) {
        // copy the agg to intermediate_agg
        intermediate_agg = agg_;
    }
    if (num_unagg != 0) {
        // Assign all left points
        exec->run(amgx_pgm::make_assign_to_exist_agg(
            weight_mtx.get(), diag.get(), agg_, intermediate_agg));
    }
    IndexType num_agg = 0;
    // Renumber the index
    exec->run(amgx_pgm::make_renumber(agg_, &num_agg));

    gko::dim<2>::dimension_type coarse_dim = num_agg;
    auto fine_dim = system_matrix_->get_size()[0];
    // prolong_row_gather is the lightway implementation for prolongation
    // TODO: However, we still create the csr to process coarse/restrict matrix
    // generation. It may be changed when we have the direct triple product from
    // agg index.
    auto prolong_row_gather = share(matrix::RowGatherer<IndexType>::create(
        exec, gko::dim<2>{fine_dim, coarse_dim}));
    exec->copy_from(exec.get(), agg_.get_num_elems(), agg_.get_const_data(),
                    prolong_row_gather->get_row_idxs());
    auto prolong_csr = share(
        csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim}, fine_dim));
    exec->copy_from(exec.get(), agg_.get_num_elems(), agg_.get_const_data(),
                    prolong_csr->get_col_idxs());
    exec->run(amgx_pgm::make_fill_seq_array(prolong_csr->get_row_ptrs(),
                                            fine_dim + 1));
    exec->run(amgx_pgm::make_fill_array(prolong_csr->get_values(), fine_dim,
                                        one<ValueType>()));
    // TODO: implement the restrict_op from aggregation.
    auto restrict_op = gko::as<csr_type>(share(prolong_csr->transpose()));
    auto restrict_sparsity =
        share(matrix::SparsityCsr<ValueType, IndexType>::create(
            exec, restrict_op->get_size(),
            restrict_op->get_num_stored_elements()));
    exec->copy_from(exec.get(), static_cast<size_type>(num_agg + 1),
                    restrict_op->get_const_row_ptrs(),
                    restrict_sparsity->get_row_ptrs());
    exec->copy_from(exec.get(), restrict_op->get_num_stored_elements(),
                    restrict_op->get_const_col_idxs(),
                    restrict_sparsity->get_col_idxs());

    // Construct the coarse matrix
    // TODO: use less memory footprint to improve it
    auto coarse_matrix =
        share(csr_type::create(exec, gko::dim<2>{coarse_dim, coarse_dim}));
    auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim});
    amgxpgm_op->apply(prolong_csr.get(), tmp.get());
    restrict_op->apply(tmp.get(), coarse_matrix.get());

    this->set_multigrid_level(prolong_row_gather, coarse_matrix,
                              restrict_sparsity);
}


#define GKO_DECLARE_AMGX_PGM(_vtype, _itype) class AmgxPgm<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_AMGX_PGM);


}  // namespace multigrid
}  // namespace gko

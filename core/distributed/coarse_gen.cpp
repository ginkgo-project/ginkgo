/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/distributed/coarse_gen.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/distributed/coarse_gen_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/amgx_pgm_kernels.hpp"


namespace gko {
namespace distributed {
namespace coarse_gen {
namespace {


GKO_REGISTER_OPERATION(match_edge, amgx_pgm::match_edge);
GKO_REGISTER_OPERATION(count_unagg, amgx_pgm::count_unagg);
GKO_REGISTER_OPERATION(renumber, amgx_pgm::renumber);
GKO_REGISTER_OPERATION(find_strongest_neighbor,
                       coarse_gen::find_strongest_neighbor);
GKO_REGISTER_OPERATION(assign_to_exist_agg, coarse_gen::assign_to_exist_agg);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // anonymous namespace
}  // namespace coarse_gen


template <typename ValueType, typename IndexType>
void CoarseGen<ValueType, IndexType>::generate_with_aggregation()
{
    using real_type = remove_complex<ValueType>;
    using matrix_type = distributed::Matrix<ValueType, IndexType>;
    using csr = matrix::Csr<ValueType, IndexType>;
    using real_csr = matrix::Csr<real_type, IndexType>;
    using weight_matrix_type = remove_complex<matrix_type>;
    auto exec = this->get_executor();
    const matrix_type* coarsegen_op =
        dynamic_cast<const matrix_type*>(system_matrix_.get());
    std::shared_ptr<const matrix_type> coarsegen_op_unique_ptr{};

    const auto global_num_rows = this->system_matrix_->get_size()[0];
    const auto local_num_rows = coarsegen_op->get_local_matrix()->get_size()[0];
    auto agg_ = coarse_indices_map_;
    Array<IndexType> strongest_neighbor(this->get_executor(), local_num_rows);
    Array<IndexType> intermediate_agg(
        this->get_executor(), parameters_.deterministic * local_num_rows);

    // Initial agg = -1
    exec->run(coarse_gen::make_fill_array(agg_.get_data(), agg_.get_num_elems(),
                                          -one<IndexType>()));
    IndexType num_unagg = local_num_rows;
    IndexType num_unagg_prev = local_num_rows;
    std::shared_ptr<weight_matrix_type> weight_mtx;
    if (parameters_.hermitian) {
        // FIXME
        // weight_mtx = coarsegen_op->compute_absolute();
    } else {
        GKO_NOT_IMPLEMENTED;
        // TODO: if mtx is a hermitian matrix, weight_mtx = abs(mtx)
        // compute weight_mtx = (abs(mtx) + abs(mtx'))/2;
        // auto abs_mtx = coarsegen_op->compute_absolute();
        // // abs_mtx is already real valuetype, so transpose is enough
        // weight_mtx = gko::as<weight_matrix_type>(abs_mtx->transpose());
        // auto half_scalar = initialize<matrix::Dense<real_type>>({0.5}, exec);
        // auto identity =
        //     matrix::Identity<real_type>::create(exec, local_num_rows);
        // // W = (abs_mtx + transpose(abs_mtx))/2
        // abs_mtx->apply(lend(half_scalar), lend(identity), lend(half_scalar),
        //                lend(weight_mtx));
    }
    // Extract the diagonal value of matrix
    auto diag =
        as<real_csr>(weight_mtx->get_local_matrix())->extract_diagonal();
    for (int i = 0; i < parameters_.max_iterations; i++) {
        // Find the strongest neighbor of each row
        exec->run(coarse_gen::make_find_strongest_neighbor(
            as<real_csr>(weight_mtx->get_local_matrix().get()),
            as<real_csr>(weight_mtx->get_non_local_matrix().get()), diag.get(),
            agg_, strongest_neighbor));
        // Match edges
        exec->run(coarse_gen::make_match_edge(strongest_neighbor, agg_));
        // Get the num_unagg
        exec->run(coarse_gen::make_count_unagg(agg_, &num_unagg));
        // no new match, all match, or the ratio of num_unagg/num is lower
        // than parameter.max_unassigned_ratio
        if (num_unagg == 0 || num_unagg == num_unagg_prev ||
            num_unagg < parameters_.max_unassigned_ratio * local_num_rows) {
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
        exec->run(coarse_gen::make_assign_to_exist_agg(
            as<real_csr>(weight_mtx->get_local_matrix().get()),
            as<real_csr>(weight_mtx->get_non_local_matrix().get()), diag.get(),
            agg_, intermediate_agg));
    }
    IndexType num_agg = 0;
    // Renumber the index
    exec->run(coarse_gen::make_renumber(agg_, &num_agg));

    gko::dim<2>::dimension_type coarse_dim = num_agg;
    auto fine_dim = system_matrix_->get_size()[0];
    // prolong_row_gather is the lightway implementation for prolongation
    // TODO: However, we still create the csr to process coarse/restrict matrix
    // generation. It may be changed when we have the direct triple product from
    // agg index.
    // auto prolong_row_gather = share(matrix::RowGatherer<IndexType>::create(
    //     exec, gko::dim<2>{fine_dim, coarse_dim}));
    // exec->copy_from(exec.get(), agg_.get_num_elems(), agg_.get_const_data(),
    //                 prolong_row_gather->get_row_idxs());
    // auto prolong_csr = share(
    //     csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim}, fine_dim));
    // exec->copy_from(exec.get(), agg_.get_num_elems(), agg_.get_const_data(),
    //                 prolong_csr->get_col_idxs());
    // exec->run(amgx_pgm::make_fill_seq_array(prolong_csr->get_row_ptrs(),
    //                                         fine_dim + 1));
    // exec->run(amgx_pgm::make_fill_array(prolong_csr->get_values(), fine_dim,
    //                                     one<ValueType>()));
    // // TODO: implement the restrict_op from aggregation.
    // auto restrict_op = gko::as<csr_type>(share(prolong_csr->transpose()));
    // auto restrict_sparsity =
    //     share(matrix::SparsityCsr<ValueType, IndexType>::create(
    //         exec, restrict_op->get_size(),
    //         restrict_op->get_num_stored_elements()));
    // exec->copy_from(exec.get(), static_cast<size_type>(num_agg + 1),
    //                 restrict_op->get_const_row_ptrs(),
    //                 restrict_sparsity->get_row_ptrs());
    // exec->copy_from(exec.get(), restrict_op->get_num_stored_elements(),
    //                 restrict_op->get_const_col_idxs(),
    //                 restrict_sparsity->get_col_idxs());

    // // Construct the coarse matrix
    // // TODO: use less memory footprint to improve it
    // auto coarse_matrix =
    //     share(csr_type::create(exec, gko::dim<2>{coarse_dim, coarse_dim}));
    // auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim});
    // amgxpgm_op->apply(prolong_csr.get(), tmp.get());
    // restrict_op->apply(tmp.get(), coarse_matrix.get());

    // this->set_multigrid_level(prolong_row_gather, coarse_matrix,
    //                           restrict_sparsity);
}


template <typename ValueType, typename IndexType>
void CoarseGen<ValueType, IndexType>::generate_with_selection()
{
    using matrix_type = distributed::Matrix<ValueType, IndexType>;
    using csr = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    using weight_matrix_type = remove_complex<matrix_type>;
    auto exec = this->get_executor();
    const matrix_type* dist_mat =
        dynamic_cast<const matrix_type*>(system_matrix_.get());

    const auto global_num_rows = dist_mat->get_size()[0];
    const auto local_num_rows = dist_mat->get_local_matrix()->get_size()[0];
}


#define GKO_DECLARE_COARSE_GEN(_vtype, _itype) class CoarseGen<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COARSE_GEN);


}  // namespace distributed
}  // namespace gko

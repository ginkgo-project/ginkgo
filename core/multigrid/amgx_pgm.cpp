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


#include "core/components/fill_array.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/amgx_pgm_kernels.hpp"


namespace gko {
namespace multigrid {
namespace amgx_pgm {


GKO_REGISTER_OPERATION(match_edge, amgx_pgm::match_edge);
GKO_REGISTER_OPERATION(count_unagg, amgx_pgm::count_unagg);
GKO_REGISTER_OPERATION(renumber, amgx_pgm::renumber);
GKO_REGISTER_OPERATION(find_strongest_neighbor,
                       amgx_pgm::find_strongest_neighbor);
GKO_REGISTER_OPERATION(assign_to_exist_agg, amgx_pgm::assign_to_exist_agg);
GKO_REGISTER_OPERATION(amgx_pgm_generate, amgx_pgm::amgx_pgm_generate);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // namespace amgx_pgm


namespace {


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> amgx_pgm_generate(
    std::shared_ptr<const Executor> exec,
    const matrix::Csr<ValueType, IndexType> *source, const size_type num_agg,
    const Array<IndexType> &agg)
{
    auto coarse = matrix::Csr<ValueType, IndexType>::create(
        exec, dim<2>{num_agg, num_agg}, 0, source->get_strategy());
    exec->run(amgx_pgm::make_amgx_pgm_generate(source, agg, coarse.get()));
    return std::move(coarse);
}


}  // namespace


template <typename ValueType, typename IndexType>
void AmgxPgm<ValueType, IndexType>::generate()
{
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    using weight_matrix_type = remove_complex<matrix_type>;
    auto exec = this->get_executor();
    const auto num_rows = this->system_matrix_->get_size()[0];
    Array<IndexType> strongest_neighbor(this->get_executor(), num_rows);
    Array<IndexType> intermediate_agg(this->get_executor(),
                                      parameters_.deterministic * num_rows);
    // Only support csr matrix currently.
    const matrix_type *amgxpgm_op = nullptr;
    // Store the csr matrix if needed
    auto amgxpgm_op_unique_ptr = matrix_type::create(exec);
    amgxpgm_op = dynamic_cast<const matrix_type *>(system_matrix_.get());
    if (!amgxpgm_op) {
        // if original matrix is not csr, converting it to csr.
        as<ConvertibleTo<matrix_type>>(this->system_matrix_.get())
            ->convert_to(amgxpgm_op_unique_ptr.get());
        amgxpgm_op = amgxpgm_op_unique_ptr.get();
    }

    // Initial agg = -1
    exec->run(amgx_pgm::make_fill_array(agg_.get_data(), agg_.get_num_elems(),
                                        -one<IndexType>()));
    IndexType num_unagg{0};
    IndexType num_unagg_prev{0};
    // TODO: if mtx is a hermitian matrix, weight_mtx = abs(mtx)
    // compute weight_mtx = (abs(mtx) + abs(mtx'))/2;
    auto abs_mtx = amgxpgm_op->compute_absolute();
    // abs_mtx is already real valuetype, so transpose is enough
    auto weight_mtx = gko::as<weight_matrix_type>(abs_mtx->transpose());
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
    // Assign all left points
    exec->run(amgx_pgm::make_assign_to_exist_agg(weight_mtx.get(), diag.get(),
                                                 agg_, intermediate_agg));
    IndexType num_agg = 0;
    // Renumber the index
    exec->run(amgx_pgm::make_renumber(agg_, &num_agg));

    // Construct the coarse matrix
    auto coarse_matrix =
        share(amgx_pgm_generate(exec, amgxpgm_op, num_agg, agg_));
    // this->set_multigrid_level(system_matrix_, coarse_matrix);
    auto coarse_dim = coarse_matrix->get_size()[0];
    auto fine_dim = system_matrix_->get_size()[0];

    // TODO: prolong_op can be done with lightway format
    auto prolong_op = share(
        matrix_type::create(exec, gko::dim<2>{fine_dim, coarse_dim}, fine_dim));
    exec->copy_from(exec.get(), agg_.get_num_elems(), agg_.get_const_data(),
                    prolong_op->get_col_idxs());
    exec->run(amgx_pgm::make_fill_seq_array(prolong_op->get_row_ptrs(),
                                            fine_dim + 1));
    exec->run(amgx_pgm::make_fill_array(prolong_op->get_values(), fine_dim,
                                        one<ValueType>()));
    // TODO: implement the restrict_op from aggregation.
    auto restrict_op = share(prolong_op->transpose());
    this->set_multigrid_level(prolong_op, coarse_matrix, restrict_op);
}


#define GKO_DECLARE_AMGX_PGM(_vtype, _itype) class AmgxPgm<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_AMGX_PGM);


}  // namespace multigrid
}  // namespace gko

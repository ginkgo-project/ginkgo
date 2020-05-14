/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/solver/multigrid.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace solver {

template <typename ValueType>
void Multigrid<ValueType>::generate()
{
    // generate coarse matrix until reaching max_level or min_coarse_rows
    auto num_rows = system_matrix_->get_size()[0];
    size_type level = 0;
    auto matrix = system_matrix_;
    while (level < parameters_.max_levels &&
           num_rows > parameters_.min_coarse_rows) {
        auto rstr_prlg_factory =
            parameters_.rstr_prlg.at(rstr_prlg_index_(num_rows, level));
        // pre_smooth_generate
        if (!pre_smoother_is_identity) {
            pre_smoother_list_.emplace_back(
                give(parameters_.pre_smoother->generate(matrix)));
        }
        // post_smooth_generate
        if (!post_smoother_is_identity) {
            pre_smoother_list_.emplace_back(
                give(parameters_.pre_smoother->generate(matrix)));
        }
        // coarse generate
        rstr_prlg_list_.emplace_back(give(factory->generate(matrix)));
        matrix = rstr_prlg_list_.back();
        num_rows = matrix->get_size()[0];
        level++;
    }
    // generate coarsest solver
    coarsest_solver =
        parameters.coarsest_solver->genreate(rstr_prlg_list_.back());
}


template <typename ValueType>
void Multigrid<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{}


template <typename ValueType>
void Multigrid<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                      const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_MULTIGRID(_type) class Multigrid<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID);


}  // namespace solver
}  // namespace gko

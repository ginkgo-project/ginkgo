/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/reorder/scaled_reordered.hpp>


#include <utility>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace experimental {
namespace reorder {


template <typename ValueType, typename IndexType>
void ScaledReordered<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                       LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            auto exec = this->get_executor();
            this->set_cache_to(dense_b, dense_x);

            // Preprocess the input vectors before applying the inner operator.
            if (row_scaling_) {
                row_scaling_->apply(cache_.inner_b.get(),
                                    cache_.intermediate.get());
                std::swap(cache_.inner_b, cache_.intermediate);
            }
            // Col scaling for x is only necessary if the inner operator uses an
            // initial guess. Otherwise x is overwritten anyway.
            if (col_scaling_ && inner_operator_->apply_uses_initial_guess()) {
                col_scaling_->inverse_apply(cache_.inner_x.get(),
                                            cache_.intermediate.get());
                std::swap(cache_.inner_x, cache_.intermediate);
            }
            if (permutation_array_.get_num_elems() > 0) {
                cache_.inner_b->row_permute(&permutation_array_,
                                            cache_.intermediate.get());
                std::swap(cache_.inner_b, cache_.intermediate);
                if (inner_operator_->apply_uses_initial_guess()) {
                    cache_.inner_x->row_permute(&permutation_array_,
                                                cache_.intermediate.get());
                    std::swap(cache_.inner_x, cache_.intermediate);
                }
            }

            inner_operator_->apply(cache_.inner_b.get(), cache_.inner_x.get());

            // Permute and scale the solution vector back.
            if (permutation_array_.get_num_elems() > 0) {
                cache_.inner_x->inverse_row_permute(&permutation_array_,
                                                    cache_.intermediate.get());
                std::swap(cache_.inner_x, cache_.intermediate);
            }
            if (col_scaling_) {
                col_scaling_->apply(cache_.inner_x.get(),
                                    cache_.intermediate.get());
                std::swap(cache_.inner_x, cache_.intermediate);
            }
            dense_x->copy_from(cache_.inner_x.get());
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void ScaledReordered<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                       const LinOp* b,
                                                       const LinOp* beta,
                                                       LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_SCALED_REORDERED(ValueType, IndexType) \
    class ScaledReordered<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCALED_REORDERED);


}  // namespace reorder
}  // namespace experimental
}  // namespace gko

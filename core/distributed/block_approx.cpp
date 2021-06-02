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

#include <ginkgo/core/distributed/block_approx.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace distributed {


template <typename ValueType, typename LocalIndexType>
void BlockApprox<ValueType, LocalIndexType>::generate(
    const Array<size_type> &block_sizes,
    const Overlap<size_type> &block_overlaps,
    const Matrix<ValueType, LocalIndexType> *matrix)
{
    auto num_blocks = block_sizes.get_num_elems();
    auto block_mtxs = matrix->get_block_approx(block_overlaps, block_sizes);
    GKO_ASSERT(block_mtxs[0]->get_executor() != nullptr);
    for (size_type j = 0; j < block_mtxs.size(); ++j) {
        diagonal_blocks_.emplace_back(std::move(block_mtxs[j]));
        block_dims_.emplace_back(diagonal_blocks_.back()->get_size());
        block_nnzs_.emplace_back(
            diagonal_blocks_.back()->get_num_stored_elements());
    }
}


template <typename ValueType, typename LocalIndexType>
void BlockApprox<ValueType, LocalIndexType>::apply_impl(const LinOp *b,
                                                        LinOp *x) const
{
    using value_type = typename MatrixType::value_type;
    using Dense = matrix::Dense<value_type>;

    // Get local block
    auto dense_b = as<Dense>(b);
    auto dense_x = as<Dense>(x);
    GKO_ASSERT(diagonal_blocks_.size() == 1);
    for (auto i = 0; i < diagonal_blocks_.size(); ++i) {
        this->diagonal_blocks_[i]->apply(dense_b, dense_x);
    }
}


template <typename ValueType, typename LocalIndexType>
void BlockApprox<ValueType, LocalIndexType>::apply_impl(const LinOp *alpha,
                                                        const LinOp *b,
                                                        const LinOp *beta,
                                                        LinOp *x) const
{
    using value_type = typename MatrixType::value_type;
    using Dense = matrix::Dense<value_type>;

    auto dense_b = as<Dense>(b);
    auto dense_x = as<Dense>(x);
    GKO_ASSERT(diagonal_blocks_.size() == 1);
    for (auto i = 0; i < diagonal_blocks_.size(); ++i) {
        this->diagonal_blocks_[i]->apply(alpha, dense_b, beta, dense_x);
    }
}


#define GKO_DECLARE_BLOCK_APPROX_CSR_GENERATE(ValueType, LocalIndexType) \
    void BlockApprox<ValueType, LocalIndexType>::generate(               \
        const Array<size_type> &, const Overlap<size_type> &,            \
        const Matrix<ValueType, LocalIndexType> *x)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_APPROX_CSR_GENERATE);


#define GKO_DECLARE_BLOCK_APPROX_CSR_APPLY(ValueType, LocalIndexType)       \
    void BlockApprox<ValueType, LocalIndexType>::apply_impl(const LinOp *b, \
                                                            LinOp *x) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_APPROX_CSR_APPLY);


#define GKO_DECLARE_BLOCK_APPROX_CSR_APPLY2(ValueType, LocalIndexType) \
    void BlockApprox<ValueType, LocalIndexType>::apply_impl(           \
        const LinOp *alpha, const LinOp *b, const LinOp *beta, LinOp *x) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_APPROX_CSR_APPLY2);


}  // namespace distributed
}  // namespace gko

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

#ifndef GKO_PUBLIC_CORE_MATRIX_BLOCK_APPROX_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BLOCK_APPROX_HPP_


#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


template <class ConcreteType>
class BlockApprox
    : public EnableLinOp<BlockApprox<ConcreteType>>,
      public EnableCreateMethod<BlockApprox<ConcreteType>>,
      public ReadableFromMatrixData<typename ConcreteType::value_type,
                                    typename ConcreteType::index_type> {
    friend class EnableCreateMethod<BlockApprox>;
    friend class EnablePolymorphicObject<BlockApprox, LinOp>;

public:
    using value_type = typename ConcreteType::value_type;
    using index_type = typename ConcreteType::index_type;
    void read(const matrix_data<value_type, index_type> &data) override {}

    size_type get_num_blocks() const { return block_mtxs_.size(); }

    std::vector<dim<2>> get_block_dimensions() const { return block_dims_; }

    std::vector<size_type> get_block_nonzeros() const { return block_nnzs_; }

    std::vector<std::shared_ptr<ConcreteType>> get_block_mtxs() const
    {
        return block_mtxs_;
    }

protected:
    BlockApprox(std::shared_ptr<const Executor> exec,
                const Array<size_type> num_blocks = {})
        : EnableLinOp<BlockApprox<ConcreteType>>{exec, dim<2>{}}, block_mtxs_{}
    {}

    BlockApprox(std::shared_ptr<const Executor> exec,
                const Array<size_type> num_blocks, const ConcreteType *matrix)
        : EnableLinOp<BlockApprox<ConcreteType>>{exec, matrix->get_size()},
          block_mtxs_{}
    {
        auto block_mtxs = matrix->get_block_approx(num_blocks);

        for (size_type j = 0; j < block_mtxs.size(); ++j) {
            block_mtxs_.emplace_back(std::move(block_mtxs[j]));
            block_dims_.emplace_back(block_mtxs_.back()->get_size());
            block_nnzs_.emplace_back(
                block_mtxs_.back()->get_num_stored_elements());
        }
    }


    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    std::vector<size_type> overlap_;
    std::vector<std::shared_ptr<ConcreteType>> block_mtxs_;
    std::vector<dim<2>> block_dims_;
    std::vector<size_type> block_nnzs_;
};


}  // namespace matrix
}  // namespace gko


#endif

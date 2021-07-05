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


template <class MatrixType>
class BlockApprox
    : public EnableLinOp<BlockApprox<MatrixType>>,
      public EnableCreateMethod<BlockApprox<MatrixType>>,
      public ReadableFromMatrixData<typename MatrixType::value_type,
                                    typename MatrixType::index_type> {
    friend class EnableCreateMethod<BlockApprox>;
    friend class EnablePolymorphicObject<BlockApprox, LinOp>;

public:
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    void read(const matrix_data<value_type, index_type> &data) override {}

    size_type get_num_blocks() const { return block_mtxs_.size(); }

    std::vector<dim<2>> get_block_dimensions() const { return block_dims_; }

    std::vector<size_type> get_block_nonzeros() const { return block_nnzs_; }

    const index_type *get_block_ptrs() const
    {
        return block_ptrs_.get_const_data();
    }

    const Overlap<size_type> &get_overlaps() const { return block_overlaps_; }

    std::vector<std::shared_ptr<MatrixType>> get_overlap_mtxs() const
    {
        return overlap_mtxs_;
    }

    std::vector<std::shared_ptr<MatrixType>> get_block_mtxs() const
    {
        return block_mtxs_;
    }

protected:
    BlockApprox(std::shared_ptr<const Executor> exec,
                const Array<size_type> &num_blocks = {},
                const Overlap<size_type> &block_overlaps = {})
        : EnableLinOp<BlockApprox<MatrixType>>{exec, dim<2>{}},
          block_overlaps_{block_overlaps},
          block_ptrs_{Array<index_type>(exec, num_blocks.get_num_elems() + 1)},
          block_mtxs_{},
          overlap_mtxs_{}
    {}

    BlockApprox(std::shared_ptr<const Executor> exec, const MatrixType *matrix,
                const Array<size_type> &num_blocks = {},
                const Overlap<size_type> &block_overlaps = {})
        : EnableLinOp<BlockApprox<MatrixType>>{exec, matrix->get_size()},
          block_overlaps_{block_overlaps},
          block_ptrs_{Array<index_type>(exec, num_blocks.get_num_elems() + 1)},
          block_mtxs_{},
          overlap_mtxs_{}
    {
        this->generate(num_blocks, block_overlaps, matrix);
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    void generate(const Array<size_type> &num_blocks,
                  const Overlap<size_type> &block_overlaps,
                  const MatrixType *matrix);

private:
    Overlap<size_type> block_overlaps_;
    std::vector<dim<2>> block_dims_;
    Array<index_type> block_ptrs_;
    std::vector<size_type> block_nnzs_;
    std::vector<std::shared_ptr<MatrixType>> overlap_mtxs_;
    std::vector<std::shared_ptr<MatrixType>> block_mtxs_;
};


}  // namespace matrix
}  // namespace gko


#endif

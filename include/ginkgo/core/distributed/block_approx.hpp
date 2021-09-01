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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_BLOCK_APPROX_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_BLOCK_APPROX_HPP_


#include <ginkgo/config.hpp>


#if GKO_HAVE_MPI


#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>


namespace gko {
namespace distributed {

template <typename ValueType = double, typename LocalIndexType = int32>
class BlockApprox
    : public EnableLinOp<BlockApprox<ValueType, LocalIndexType>>,
      public EnableCreateMethod<BlockApprox<ValueType, LocalIndexType>>,
      public DistributedBase {
    friend class EnableCreateMethod<BlockApprox>;
    friend class EnablePolymorphicObject<BlockApprox, LinOp>;

public:
    using value_type = ValueType;
    using index_type = global_index_type;
    using local_index_type = LocalIndexType;
    using MatrixType = typename Matrix<ValueType, LocalIndexType>::LocalMtx;

    size_type get_num_blocks() const { return diagonal_blocks_.size(); }

    std::vector<dim<2>> get_block_dimensions() const { return block_dims_; }

    std::vector<size_type> get_block_nonzeros() const { return block_nnzs_; }

    const Overlap<size_type> &get_overlaps() const { return block_overlaps_; }

    std::vector<std::shared_ptr<MatrixType>> get_block_mtxs() const
    {
        return diagonal_blocks_;
    }

protected:
    BlockApprox(std::shared_ptr<const Executor> exec,
                std::shared_ptr<mpi::communicator> comm =
                    std::make_shared<mpi::communicator>(),
                const Array<size_type> &num_blocks = {},
                const Overlap<size_type> &block_overlaps = {})
        : EnableLinOp<BlockApprox<value_type, local_index_type>>{exec,
                                                                 dim<2>{}},
          DistributedBase{comm},
          block_overlaps_{block_overlaps},
          diagonal_blocks_{}
    {}

    BlockApprox(std::shared_ptr<const Executor> exec,
                const Matrix<value_type, local_index_type> *matrix,
                std::shared_ptr<mpi::communicator> comm,
                const Array<size_type> &num_blocks = {},
                const Overlap<size_type> &block_overlaps = {})
        : EnableLinOp<
              BlockApprox<value_type, local_index_type>>{exec,
                                                         matrix->get_size()},
          DistributedBase{comm},
          block_overlaps_{block_overlaps},
          diagonal_blocks_{}
    {
        this->generate(num_blocks, block_overlaps, matrix);
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    void generate(const Array<size_type> &num_blocks,
                  const Overlap<size_type> &block_overlaps,
                  const Matrix<value_type, local_index_type> *matrix);

private:
    Overlap<size_type> block_overlaps_;
    std::vector<dim<2>> block_dims_;
    std::vector<size_type> block_nnzs_;
    std::vector<std::shared_ptr<MatrixType>> diagonal_blocks_;
};


}  // namespace distributed
}  // namespace gko


#else


namespace gko {
namespace distributed {
template <typename ValueType, typename IndexType>
class BlockApprox;
}
}  // namespace gko


#endif  // GKO_HAVE_MPI
#endif

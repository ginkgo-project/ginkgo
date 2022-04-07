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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_


#include <vector>


#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/identity.hpp>


namespace gko {
namespace matrix {


/**
 * BatchIdentity is a batch matrix format which explicitly stores all values of
 * the matrix in each of the batches.
 *
 * The values in each of the batches are stored in row-major format (values
 * belonging to the same row appear consecutive in the memory). Optionally, rows
 * can be padded for better memory access.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @note While this format is not very useful for storing sparse matrices, it
 *       is often suitable to store vectors, and sets of vectors.
 * @ingroup batch_dense
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchIdentity : public EnableBatchLinOp<BatchIdentity<ValueType>>,
                      public EnableCreateMethod<BatchIdentity<ValueType>>,
                      public BatchTransposable {
    friend class EnableCreateMethod<BatchIdentity>;
    friend class EnablePolymorphicObject<BatchIdentity, BatchLinOp>;

public:
    using EnableBatchLinOp<BatchIdentity>::convert_to;
    using EnableBatchLinOp<BatchIdentity>::move_to;

    using value_type = ValueType;
    using index_type = int32;
    using transposed_type = BatchIdentity<ValueType>;
    using unbatch_type = Identity<ValueType>;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<BatchIdentity>;
    using complex_type = to_complex<BatchIdentity>;

    std::unique_ptr<BatchLinOp> transpose() const override
    {
        return this->clone();
    }

    std::unique_ptr<BatchLinOp> conj_transpose() const override
    {
        return this->clone();
    }

    /**
     * Unbatches the batched dense and creates a std::vector of Identity
     * matrices
     *
     * @return  a std::vector containing the Identity matrices.
     */
    std::vector<std::unique_ptr<unbatch_type>> unbatch() const
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(this);
        auto exec = this->get_executor();
        auto unbatch_mats = std::vector<std::unique_ptr<unbatch_type>>{};
        const auto c_entry_size =
            std::min(this->get_size().at(0)[0], this->get_size().at(0)[1]);
        for (size_type b = 0; b < this->get_num_batch_entries(); ++b) {
            auto mat = unbatch_type::create(exec, this->get_size().at(b)[0]);
            unbatch_mats.emplace_back(std::move(mat));
        }
        return unbatch_mats;
    }

private:
    /**
     * Extract sizes from the vector of the distinct Identity matrices.
     */
    batch_dim<2> get_sizes_from_mtxs(
        const std::vector<Identity<ValueType>*> mtxs) const
    {
        auto sizes = std::vector<dim<2>>(mtxs.size());
        for (auto i = 0; i < mtxs.size(); ++i) {
            sizes[i] = mtxs[i]->get_size();
        }
        return batch_dim<2>(sizes);
    }

protected:
    /**
     * Creates an uninitialized BatchIdentity matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the batch matrices in a batch_dim object
     */
    BatchIdentity(std::shared_ptr<const Executor> exec,
                  const batch_dim<2>& size = batch_dim<2>{})
        : EnableBatchLinOp<BatchIdentity>(exec, size)
    {}

    /**
     * Creates a BatchIdentity matrix from a vector of matrices
     *
     * @param exec  Executor associated to the matrix
     * @param matrices  The matrices that need to be batched.
     */
    BatchIdentity(std::shared_ptr<const Executor> exec,
                  const std::vector<Identity<ValueType>*>& matrices)
        : EnableBatchLinOp<BatchIdentity>(exec, get_sizes_from_mtxs(matrices))
    {}

    /**
     * Creates a BatchIdentity matrix by duplicating BatchIdentity matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  The number of times to duplicate
     * @param input  The matrix to be duplicated.
     */
    BatchIdentity(std::shared_ptr<const Executor> exec,
                  const size_type num_duplications,
                  const BatchIdentity<value_type>* const input)
        : EnableBatchLinOp<BatchIdentity>(exec)
    {
        const auto nbatch1 = input->get_num_batch_entries();
        if (input->get_size().stores_equal_sizes()) {
            this->set_size(batch_dim<2>(nbatch1 * num_duplications,
                                        input->get_size().at(0)));
        } else {
            std::vector<dim<2>> sizes(nbatch1 * num_duplications);
            for (size_type idup = 0; idup < num_duplications; idup++) {
                for (size_type i = 0; i < nbatch1; i++) {
                    sizes[idup * nbatch1 + i] = input->get_size().at(i);
                }
            }
            this->set_size(sizes);
        }
    }

    /**
     * Creates a BatchIdentity matrix by duplicating Identity matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  The number of times to duplicate
     * @param input  The matrix to be duplicated.
     */
    BatchIdentity(std::shared_ptr<const Executor> exec,
                  const size_type num_duplications,
                  const Identity<value_type>* const input)
        : EnableBatchLinOp<BatchIdentity>(
              exec, gko::batch_dim<2>(num_duplications, input->get_size()))
    {}

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override
    {
        x->copy_from(b);
    }

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta,
                    BatchLinOp* x) const override GKO_NOT_IMPLEMENTED;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_

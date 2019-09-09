/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_REORDER_METIS_FILL_REDUCE_HPP_
#define GKO_CORE_REORDER_METIS_FILL_REDUCE_HPP_


#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/reorder/reordering.hpp>


namespace gko {
/**
 * @brief The Reorder namespace.
 *
 * @ingroup reorder
 */
namespace reorder {


/**
 * ParILU is an incomplete LU reorder which is computed in parallel.
 *
 * $L$ is a lower unitriangular, while $U$ is an upper triangular matrix, which
 * approximate a given matrix $A$ with $A \approx LU$. Here, $L$ and $U$ have
 * the same sparsity pattern as $A$, which is also called ILU(0).
 *
 * The ParILU algorithm generates the incomplete factors iteratively, using a
 * fixed-point iteration of the form
 *
 * $
 * F(L, U) =
 * \begin{cases}
 *     \frac{1}{u_{jj}}
 *         \left(a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj}\right), \quad & i>j \\
 *     a_{ij}-\sum_{k=1}^{i-1}l_{ik}u_{kj}, \quad & i\leq j
 * \end{cases}
 * $
 *
 * In general, the entries of $L$ and $U$ can be iterated in parallel and in
 * asynchronous fashion, the algorithm asymptotically converges to the
 * incomplete factors $L$ and $U$ fulfilling $\left(R = A - L \cdot
 * U\right)\vert_\mathcal{S} = 0\vert_\mathcal{S}$ where $\mathcal{S}$ is the
 * pre-defined sparsity pattern (in case of ILU(0) the sparsity pattern of the
 * system matrix $A$). The number of ParILU sweeps needed for convergence
 * depends on the parallelism level: For sequential execution, a single sweep
 * is sufficient, for fine-grained parallelism, 3 sweeps are typically
 * generating a good approximation.
 *
 * The ParILU algorithm in Ginkgo follows the design of E. Chow and A. Patel,
 * Fine-grained Parallel Incomplete LU Reorder, SIAM Journal on Scientific
 * Computing, 37, C169-C193 (2015).
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup linop
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class MetisFillReduce
    : public EnablePolymorphicObject<MetisFillReduce<ValueType, IndexType>,
                                     Reordering> {
    friend class EnablePolymorphicObject<MetisFillReduce, Reordering>;

public:
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> get_system_matrix() const
    {
        return system_matrix_;
    }

    std::shared_ptr<Array<IndexType>> get_permutation() const
    {
        return permutation_;
    }

    std::shared_ptr<Array<IndexType>> get_adj_ptrs() const { return adj_ptrs_; }

    std::shared_ptr<Array<IndexType>> get_adj_idxs() const { return adj_idxs_; }

    std::shared_ptr<Array<IndexType>> get_inverse_permutation() const
    {
        return inv_permutation_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * If this parameter is set then an inverse permutation matrix is also
         * constructed along with the normal permutation matrix.
         */
        bool GKO_FACTORY_PARAMETER(remove_diagonal_elements, true);

        /**
         * If this parameter is set then an inverse permutation matrix is also
         * constructed along with the normal permutation matrix.
         */
        bool GKO_FACTORY_PARAMETER(construct_inverse_permutation, false);

        /**
         * The number of iterations the `compute` kernel will use when doing
         * the reorder. The default value `0` means `Auto`, so the
         * implementation decides on the actual value depending on the
         * resources that are available.
         */
        Array<IndexType> GKO_FACTORY_PARAMETER(vertex_weights, nullptr);
    };
    GKO_ENABLE_REORDERING_FACTORY(MetisFillReduce, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Permutes the input Linear operator.
     */
    void permute(LinOp *to_permute) const;

    /**
     * Inverts the permutation the input Linear operator.
     */
    void inverse_permute(LinOp *to_permute) const;

    /**
     * Generates the permutation matrix and if required the inverse permutation
     * matrix.
     */
    void generate() const;

    explicit MetisFillReduce(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<MetisFillReduce, Reordering>(std::move(exec))
    {}

    explicit MetisFillReduce(const Factory *factory, const ReorderingArgs &args)
        : EnablePolymorphicObject<MetisFillReduce, Reordering>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        using CsrMatrix = matrix::Csr<ValueType, IndexType>;

        const auto exec = this->get_executor();
        GKO_ASSERT_IS_SQUARE_MATRIX(args.system_matrix);
        // This is needed because it does not make sense to call the copy and
        // convert if the existing matrix is empty.
        if (!args.system_matrix->get_size()) {
            system_matrix_ = CsrMatrix::create(exec);
        } else {
            system_matrix_ =
                copy_and_convert_to<CsrMatrix>(exec, args.system_matrix);
        }
        vertex_weights_ = std::shared_ptr<Array<IndexType>>(
            new Array<IndexType>{exec, system_matrix_->get_size()[0]});
        for (auto i = 0; i < system_matrix_->get_size()[0]; ++i) {
            vertex_weights_->get_data()[i] = 1;
        }
        adj_ptrs_ = std::shared_ptr<Array<IndexType>>(
            new Array<IndexType>{exec, system_matrix_->get_size()[0] + 1});
        adj_idxs_ = std::shared_ptr<Array<IndexType>>(new Array<IndexType>{
            exec, system_matrix_->get_num_stored_elements() -
                      system_matrix_->get_size()[0]});
        permutation_ = std::shared_ptr<Array<IndexType>>(
            new Array<IndexType>{exec, system_matrix_->get_size()[0]});
        inv_permutation_ = std::shared_ptr<Array<IndexType>>(
            new Array<IndexType>{exec, system_matrix_->get_size()[0]});
        permutation_mat_ = CsrMatrix::create(exec, system_matrix_->get_size());
        inv_permutation_mat_ =
            CsrMatrix::create(exec, system_matrix_->get_size());
        this->generate();
    }

private:
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> system_matrix_{};
    std::shared_ptr<Array<IndexType>> adj_ptrs_{};
    std::shared_ptr<Array<IndexType>> adj_idxs_{};
    std::shared_ptr<Array<IndexType>> vertex_weights_{};
    std::shared_ptr<Array<IndexType>> permutation_{};
    std::shared_ptr<Array<IndexType>> inv_permutation_{};
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> permutation_mat_{};
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> inv_permutation_mat_{};
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_CORE_REORDER_METIS_FILL_REDUCE_HPP_

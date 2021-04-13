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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_RAS_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_RAS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


/**
 * A block-Ras preconditioner is a block-diagonal linear operator, obtained
 * by inverting the diagonal blocks of the source operator.
 *
 * The Ras class implements the inversion of the diagonal blocks using
 * Gauss-Jordan elimination with column pivoting, and stores the inverse
 * explicitly in a customized format.
 *
 * If the diagonal blocks of the matrix are not explicitly set by the user, the
 * implementation will try to automatically detect the blocks by first finding
 * the natural blocks of the matrix, and then applying the supervariable
 * agglomeration procedure on them. However, if problem-specific knowledge
 * regarding the block diagonal structure is available, it is usually beneficial
 * to explicitly pass the starting rows of the diagonal blocks, as the block
 * detection is merely a heuristic and cannot perfectly detect the diagonal
 * block structure. The current implementation supports blocks of up to 32 rows
 * / columns.
 *
 * The implementation also includes an improved, adaptive version of the
 * block-Ras preconditioner, which can store some of the blocks in lower
 * precision and thus improve the performance of preconditioner application by
 * reducing the amount of memory transfers. This variant can be enabled by
 * setting the Ras::Factory's `storage_optimization` parameter.  Refer to the
 * documentation of the parameter for more details.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 *
 * @note The current implementation supports blocks of up to 32 rows / columns.
 * @note When using the adaptive variant, there may be a trade-off in terms of
 *       slightly longer preconditioner generation due to extra work required to
 *       detect the optimal precision of the blocks.
 *
 * @ingroup ras
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Ras : public EnableLinOp<Ras<ValueType, IndexType>>, public Transposable {
    friend class EnableLinOp<Ras>;
    friend class EnablePolymorphicObject<Ras, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using transposed_type = Ras<ValueType, IndexType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    bool apply_uses_initial_guess() const override { return true; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Inner solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            solver, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Ras, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Creates an empty Ras preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit Ras(std::shared_ptr<const Executor> exec) : EnableLinOp<Ras>(exec)
    {}

    /**
     * Creates a Ras preconditioner from a matrix using a Ras::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit Ras(const Factory *factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Ras>(factory->get_executor(),
                           gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {
        this->generate(lend(system_matrix));
    }

    /**
     * Generates the preconditoner.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       preconditioner
     */
    void generate(const LinOp *system_matrix);

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    std::vector<std::shared_ptr<LinOp>> inner_solvers_;
    Overlap<size_type> overlaps_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_RAS_HPP_

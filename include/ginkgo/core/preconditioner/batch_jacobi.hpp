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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


/**
 * A batch-Jacobi preconditioner is a diagonal batch linear operator, obtained
 * by inverting the diagonals, of the source batch operator.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup jacobi
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BatchJacobi : public EnableBatchLinOp<BatchJacobi<ValueType, IndexType>>,
                    public BatchTransposable {
    friend class EnableBatchLinOp<BatchJacobi>;
    friend class EnablePolymorphicObject<BatchJacobi, BatchLinOp>;

public:
    using EnableBatchLinOp<BatchJacobi>::convert_to;
    using EnableBatchLinOp<BatchJacobi>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchJacobi, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    std::unique_ptr<BatchLinOp> transpose() const override
    {
        return this->clone();
    }

    std::unique_ptr<BatchLinOp> conj_transpose() const override
    {
        // Since this preconditioner does nothing in its genarate step,
        //  conjugate transpose only depends on the matrix being
        //  conjugate-transposed.
        return this->clone();
    }

protected:
    /**
     * Creates an empty Jacobi preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit BatchJacobi(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchJacobi>(exec)
    {}

    /**
     * Creates a Jacobi preconditioner from a matrix using a Jacobi::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit BatchJacobi(const Factory* factory,
                         std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchJacobi>(
              factory->get_executor(),
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
    void generate(const BatchLinOp* system_matrix) {}

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override{};

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override{};
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_

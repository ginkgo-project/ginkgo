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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_PAR_ILU_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_PAR_ILU_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>


namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


/**
 * A batch of incomplete LU factor preconditioners for a batch of matrices.
 *
 * This computes parIlu(0) preconditioner
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup ilu
 * @ingroup precond
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BatchParIlu : public EnableBatchLinOp<BatchParIlu<ValueType, IndexType>>,
                    public BatchTransposable {
    friend class EnableBatchLinOp<BatchParIlu>;
    friend class EnablePolymorphicObject<BatchParIlu, BatchLinOp>;

public:
    using EnableBatchLinOp<BatchParIlu>::convert_to;
    using EnableBatchLinOp<BatchParIlu>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::BatchCsr<ValueType, IndexType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * @brief Optimization parameter that skips the sorting of the input
         *        matrix (only skip if it is known that it is already sorted).
         *
         * The algorithm to compute parILU(0) requires the
         * input matrix to be sorted. If it is, this parameter can be set to
         * `true` to skip the sorting for better performance.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        /**
         * Number of sweeps for parIlu0
         *
         */
        int GKO_FACTORY_PARAMETER_SCALAR(num_sweeps, 10);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchParIlu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    const matrix::BatchCsr<ValueType, IndexType>* get_const_l_factor() const
    {
        return l_factor_.get();
    }

    const matrix::BatchCsr<ValueType, IndexType>* get_const_u_factor() const
    {
        return u_factor_.get();
    }

protected:
    /**
     * Creates an empty Ilu preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit BatchParIlu(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchParIlu>(exec)
    {}

    /**
     * Creates a Ilu preconditioner from a matrix using a Ilu::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit BatchParIlu(const Factory* factory,
                         std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchParIlu>(
              factory->get_executor(),
              gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {
        this->generate_precond(lend(system_matrix));
    }

    /**
     * Generates the preconditoner.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       preconditioner
     */
    void generate_precond(const BatchLinOp* system_matrix);

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override{};

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override{};

private:
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>> l_factor_;
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>> u_factor_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_PAR_ILU_HPP_

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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>


namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


enum class batch_ilu_type { parilu, exact_ilu };

/**
 * A batch of incomplete LU factor preconditioners for a batch of matrices.
 *
 * Currently, this always computes ILU(0) preconditioners
 *
 * Note: Batched Preconditioners do not support user facing apply.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup ilu
 * @ingroup precond
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BatchIlu : public EnableBatchLinOp<BatchIlu<ValueType, IndexType>>,
                 public BatchTransposable {
    friend class EnableBatchLinOp<BatchIlu>;
    friend class EnablePolymorphicObject<BatchIlu, BatchLinOp>;

public:
    using EnableBatchLinOp<BatchIlu>::convert_to;
    using EnableBatchLinOp<BatchIlu>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::BatchCsr<ValueType, IndexType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * @brief Optimization parameter that skips the sorting of the input
         *        matrix (only skip if it is known that it is already sorted).
         *
         * The algorithm to generate exact ilu0 requires the
         * input matrix to be sorted. If it is, this parameter can be set to
         * `true` to skip the sorting for better performance.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        /**
         * @brief Number of sweeps for parilu0
         *
         * This parameter is used only in case of parilu0 preconditioner.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(parilu_num_sweeps, 10);

        /**
         * @brief batch_ilu_type
         *
         */
        batch_ilu_type GKO_FACTORY_PARAMETER_SCALAR(ilu_type,
                                                    batch_ilu_type::exact_ilu);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchIlu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);


    std::unique_ptr<BatchLinOp> transpose() const override
    {
        return build()
            .with_skip_sorting(this->parameters_.skip_sorting)
            .with_ilu_type(this->parameters_.ilu_type)
            .with_parilu_num_sweeps(this->parameters_.parilu_num_sweeps)
            .on(this->get_executor())
            ->generate(share(
                as<BatchTransposable>(this->system_matrix_)->transpose()));
    }

    std::unique_ptr<BatchLinOp> conj_transpose() const override
    {
        return build()
            .with_skip_sorting(this->parameters_.skip_sorting)
            .with_ilu_type(this->parameters_.ilu_type)
            .with_parilu_num_sweeps(this->parameters_.parilu_num_sweeps)
            .on(this->get_executor())
            ->generate(share(
                as<BatchTransposable>(this->system_matrix_)->conj_transpose()));
    }

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_factorized_matrix() const
    {
        return mat_factored_;
    }

    const IndexType* get_const_diag_locations() const
    {
        return diag_locations_.get_const_data();
    }

    std::pair<std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>,
              std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>>
    generate_split_factors_from_factored_matrix() const;

protected:
    /**
     * Creates an empty Ilu preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit BatchIlu(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchIlu>(exec)
    {}

    /**
     * Creates a Ilu preconditioner from a matrix using a Ilu::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit BatchIlu(const Factory* factory,
                      std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchIlu>(factory->get_executor(),
                                     gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix}
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(system_matrix);
        this->generate_precond();
    }

    /**
     * Generates the preconditoner.
     *
     */
    void generate_precond();

    // Since there is no guarantee that the complete generation of the
    // preconditioner would occur outside the solver kernel, that is in the
    // external generate step, there is no logic in implementing "apply" for
    // batched preconditioners
    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override
        GKO_BATCHED_NOT_SUPPORTED(
            "batched preconditioners do not support (user facing/public) "
            "apply");


    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override
        GKO_BATCHED_NOT_SUPPORTED(
            "batched preconditioners do not support (user facing/public) "
            "apply");


private:
    // Note: Storing the system matrix is necessary because it is being used in
    // the transpose and conjugate transpose.
    std::shared_ptr<const BatchLinOp> system_matrix_;
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>> mat_factored_;
    array<IndexType> diag_locations_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_HPP_

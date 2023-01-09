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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/preconditioner/batch_ilu.hpp>

namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


enum class batch_ilu_isai_apply {
    spmv_isai_simple,
    spmv_isai_with_spgemm,
    relaxation_steps_isai_simple,
    relaxation_steps_isai_with_spgemm
};


/**
 * A batch of ILU-ISAI preconditioners for a batch of matrices.
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
class BatchIluIsai
    : public EnableBatchLinOp<BatchIluIsai<ValueType, IndexType>>,
      public BatchTransposable {
    friend class EnableBatchLinOp<BatchIluIsai>;
    friend class EnablePolymorphicObject<BatchIluIsai, BatchLinOp>;

public:
    using EnableBatchLinOp<BatchIluIsai>::convert_to;
    using EnableBatchLinOp<BatchIluIsai>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::BatchCsr<ValueType, IndexType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * @brief Optimization parameter that skips the sorting of the input
         *        matrix (only skip if it is known that it is already sorted).
         *
         * The algorithm to generate ilu0 and isai requires the
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
         * @brief batch_ilu_isai_type
         *
         */
        batch_ilu_type GKO_FACTORY_PARAMETER_SCALAR(ilu_type,
                                                    batch_ilu_type::exact_ilu);

        /**
         * @brief Sparisty pattern power for batch isai generation of lower
         * triangular factors.
         *
         * Since spy(A_i ^ k) for k > 1 may not be the same for all matrices in
         * the batch, the batched isai generation algorithm uses the sparisty
         * pattern resulting from the first matrix in the batch to generate
         * inverses for all the batch members.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(lower_factor_isai_sparsity_power, 1);


        /**
         * @brief Sparisty pattern power for isai generation of upper triangular
         * factors.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(upper_factor_isai_sparsity_power, 1);

        /**
         * @brief Batch ilu-isai apply type.
         *
         * There are 4 ways to apply the preconditioner (z = precond* r):
         * 1. batch_ilu_isai_apply::spmv_isai_simple -> z = (U_left_isai *
         * (L_left_isai * r)) using 2 spmvs
         * 2. batch_ilu_isai_apply::spmv_isai_with_spgemm -> z = ((U_left_isai *
         * L_left_isai) * r)  using an spgemm and an spmv
         * 3. batch_ilu_isai_apply::relaxation_steps_isai_simple -> solving L *
         * y = r and then U * z = y using isai-relaxation steps method i.e
         * y_updated = L_left_isai * r + (I - L_left_isai * L) * y_old (iterate)
         * Once y is obtained, z_updated = U_left_isai * y + (I - U_left_isai *
         * U) * z_old
         * 4. batch_ilu_isai_apply::relaxation_steps_isai_with_spgemm -> solving
         * L * y = r and then U * z = y using isai-relaxation steps method with
         * pre-computed iteration-matrices (which makes use of spgemm)
         *
         */
        batch_ilu_isai_apply GKO_FACTORY_PARAMETER_SCALAR(
            apply_type, batch_ilu_isai_apply::spmv_isai_simple);

        /**
         * @brief Number of relaxtion steps.
         *
         * This parameter is only used in case of apply_type =
         * relaxation_steps_isai_simple or relaxation_steps_isai_with_spgemm
         *
         */
        int GKO_FACTORY_PARAMETER_SCALAR(num_relaxation_steps, 3);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchIluIsai, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);


    std::unique_ptr<BatchLinOp> transpose() const override
    {
        return build()
            .with_skip_sorting(this->parameters_.skip_sorting)
            .with_ilu_type(this->parameters_.ilu_type)
            .with_parilu_num_sweeps(this->parameters_.parilu_num_sweeps)
            .with_lower_factor_isai_sparsity_power(
                this->parameters_.lower_factor_isai_sparsity_power)
            .with_upper_factor_isai_sparsity_power(
                this->parameters_.upper_factor_isai_sparsity_power)
            .with_apply_type(this->parameters_.apply_type)
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
            .with_lower_factor_isai_sparsity_power(
                this->parameters_.lower_factor_isai_sparsity_power)
            .with_upper_factor_isai_sparsity_power(
                this->parameters_.upper_factor_isai_sparsity_power)
            .with_apply_type(this->parameters_.apply_type)
            .on(this->get_executor())
            ->generate(share(
                as<BatchTransposable>(this->system_matrix_)->conj_transpose()));
    }

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_lower_factor_isai() const
    {
        return lower_factor_isai_;
    }

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_upper_factor() const
    {
        return upper_factor_;
    }

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_lower_factor() const
    {
        return lower_factor_;
    }

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_upper_factor_isai() const
    {
        return upper_factor_isai_;
    }

    batch_ilu_isai_apply get_apply_type() const
    {
        return parameters_.apply_type;
    }

    int get_num_relaxation_steps() const
    {
        return parameters_.num_relaxation_steps;
    }

    /**
     * Note: Will return null shared_ptr if apply_type != spmv_isai_with_spgemm
     *
     */
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_mult_inv() const
    {
        return mult_inv_;
    }

    /**
     * Note: Will return null shared_ptr if apply_type !=
     * relaxtion_steps_isai_with_spgemm
     *
     */
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_iteration_matrix_lower_solve() const
    {
        return iter_mat_lower_solve_;
    }

    /**
     * Note: Will return null shared_ptr if apply_type !=
     * relaxtion_steps_isai_with_spgemm
     *
     */
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_iteration_matrix_upper_solve() const
    {
        return iter_mat_upper_solve_;
    }


protected:
    /**
     * Creates an empty IluIsai preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit BatchIluIsai(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchIluIsai>(exec)
    {}

    /**
     * Creates a IluIsai preconditioner from a matrix using a IluIsai::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit BatchIluIsai(const Factory* factory,
                          std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchIluIsai>(
              factory->get_executor(),
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
    // Note: Make these two const to avoid cloning the isai precond's approx inv
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
        lower_factor_isai_;
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
        upper_factor_isai_;
    // Note: Make these two const to avoid cloning the ilu preconditioner's
    // factors generated by split function
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>> lower_factor_;
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>> upper_factor_;
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>> mult_inv_{nullptr};
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>>
        iter_mat_lower_solve_{nullptr};
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>>
        iter_mat_upper_solve_{nullptr};
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_

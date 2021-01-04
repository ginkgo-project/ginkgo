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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_ISAI_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_ISAI_HPP_


#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
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
 * This enum lists the types of the ISAI preconditioner.
 *
 * ISAI can either generate a lower triangular matrix, or an upper triangular
 * matrix.
 */
enum struct isai_type { lower, upper };

/**
 * The Incomplete Sparse Approximate Inverse (ISAI) Preconditioner generates
 * an approximate inverse matrix for a given lower triangular matrix L or upper
 * triangular matrix U.
 *
 * Using the preconditioner computes $aiU * x$ or $aiL * x$ (depending on the
 * type of the Isai) for a given vector x (may have multiple right hand sides).
 * aiU and aiL are the approximate inverses for U and L respectively.
 *
 * The sparsity pattern used for the approximate inverse is the same as
 * the sparsity pattern of the respective triangular matrix.
 *
 * For more details on the algorithm, see the paper
 * <a href="https://doi.org/10.1016/j.parco.2017.10.003">
 * Incomplete Sparse Approximate Inverses for Parallel Preconditioning</a>,
 * which is the basis for this work.
 *
 * @note GPU implementations can only handle the vector unit width `width`
 *       (warp size for CUDA) as number of elements per row in the sparse
 *       matrix. If there are more than `width` elements per row, the remaining
 *       elements will be ignored.
 *
 * @tparam IsaiType  determines if the ISAI is generated for a lower triangular
 *                   matrix or an upper triangular matrix
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup isai
 * @ingroup precond
 * @ingroup LinOp
 */
template <isai_type IsaiType, typename ValueType, typename IndexType>
class Isai : public EnableLinOp<Isai<IsaiType, ValueType, IndexType>>,
             public Transposable {
    friend class EnableLinOp<Isai>;
    friend class EnablePolymorphicObject<Isai, LinOp>;
    friend class Isai<IsaiType == isai_type::lower ? isai_type::upper
                                                   : isai_type::lower,
                      ValueType, IndexType>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type =
        Isai<IsaiType == isai_type::lower ? isai_type::upper : isai_type::lower,
             ValueType, IndexType>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    static constexpr isai_type type{IsaiType};

    /**
     * Returns the approximate inverse of the given matrix (either L or U,
     * depending on the template parameter IsaiType).
     *
     * @returns the generated approximate inverse
     */
    std::shared_ptr<const Csr> get_approximate_inverse() const
    {
        return approximate_inverse_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * @brief Optimization parameter that skips the sorting of the input
         *        matrix (only skip if it is known that it is already sorted).
         *
         * The algorithm to create the approximate inverses requires the
         * input matrix to be sorted. If it is, this parameter can be set to
         * `true` to skip the sorting for better performance.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        /**
         * @brief Which power of the input matrix should be used for the
         *        sparsity pattern.
         *
         * The algorithm symbolically computes M^n and uses this sparsity
         * pattern for the sparse inverse.
         * Must be at least 1, default value 1.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(sparsity_power, 1);
    };

    GKO_ENABLE_LIN_OP_FACTORY(Isai, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

protected:
    explicit Isai(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Isai>(std::move(exec))
    {}

    /**
     * Creates an Isai preconditioner from a matrix using an Isai::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param factors  Composition<ValueType> of a lower triangular and an
     *                 upper triangular matrix (L and U)
     */
    explicit Isai(const Factory *factory,
                  std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Isai>(factory->get_executor(), system_matrix->get_size()),
          parameters_{factory->get_parameters()}
    {
        const auto skip_sorting = parameters_.skip_sorting;
        const auto power = parameters_.sparsity_power;
        generate_inverse(system_matrix, skip_sorting, power);
    }

    void apply_impl(const LinOp *b, LinOp *x) const override
    {
        approximate_inverse_->apply(b, x);
    }

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {
        approximate_inverse_->apply(alpha, b, beta, x);
    }

private:
    /**
     * Generates the approximate inverse for a triangular matrix and
     * stores the result in `approximate_inverse_`.
     *
     * @param to_invert  the source triangular matrix used to generate
     *                     the approximate inverse
     *
     * @param skip_sorting  dictates if the sorting of the input matrix should
     *                      be skipped.
     */
    void generate_inverse(std::shared_ptr<const LinOp> to_invert,
                          bool skip_sorting, int power);

private:
    std::shared_ptr<Csr> approximate_inverse_;
};


template <typename ValueType = default_precision, typename IndexType = int32>
using LowerIsai = Isai<isai_type::lower, ValueType, IndexType>;

template <typename ValueType = default_precision, typename IndexType = int32>
using UpperIsai = Isai<isai_type::upper, ValueType, IndexType>;


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_ISAI_HPP_

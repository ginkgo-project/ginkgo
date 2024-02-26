// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
 * ISAI can either be generated for a general square matrix, a lower triangular
 * matrix, an upper triangular matrix or an spd matrix.
 */
enum struct isai_type { lower, upper, general, spd };

/**
 * The Incomplete Sparse Approximate Inverse (ISAI) Preconditioner generates
 * an approximate inverse matrix for a given square matrix A, lower triangular
 * matrix L, upper triangular matrix U or symmetric positive (spd) matrix B.
 *
 * Using the preconditioner computes $aiA * x$, $aiU * x$, $aiL * x$ or $aiC^T *
 * aiC * x$ (depending on the type of the Isai) for a given vector x (may have
 * multiple right hand sides). aiA, aiU and aiL are the approximate inverses for
 * A, U and L respectively. aiC is an approximation to C, the exact Cholesky
 * factor of B (This is commonly referred to as a Factorized Sparse Approximate
 * Inverse, short FSPAI).
 *
 * The sparsity pattern used for the approximate inverse of A, L and U is the
 * same as the sparsity pattern of the respective matrix. For B, the sparsity
 * pattern used for the approximate inverse is the same as the sparsity pattern
 * of the lower triangular half of B.
 *
 * Note that, except for the spd case, for a matrix A generally
 * ISAI(A)^T != ISAI(A^T).
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
 * @tparam IsaiType  determines if the ISAI is generated for a general square
 *         matrix, a lower triangular matrix, an upper triangular matrix or an
 *         spd matrix
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
    friend class Isai<isai_type::general, ValueType, IndexType>;
    friend class Isai<isai_type::lower, ValueType, IndexType>;
    friend class Isai<isai_type::upper, ValueType, IndexType>;
    friend class Isai<isai_type::spd, ValueType, IndexType>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type =
        Isai<IsaiType == isai_type::general ? isai_type::general
             : IsaiType == isai_type::spd   ? isai_type::spd
             : IsaiType == isai_type::lower ? isai_type::upper
                                            : isai_type::lower,
             ValueType, IndexType>;
    using Comp = Composition<ValueType>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    using Dense = matrix::Dense<ValueType>;
    static constexpr isai_type type{IsaiType};

    /**
     * Returns the approximate inverse of the given matrix (either a CSR matrix
     * for IsaiType general, upper or lower or a composition of two CSR matrices
     * for IsaiType spd).
     *
     * @returns the generated approximate inverse
     */
    std::shared_ptr<const typename std::conditional<IsaiType == isai_type::spd,
                                                    Comp, Csr>::type>
    get_approximate_inverse() const
    {
        return as<typename std::conditional<IsaiType == isai_type::spd, Comp,
                                            Csr>::type>(approximate_inverse_);
    }

    /**
     * Copy-assigns an ISAI preconditioner. Preserves the executor,
     * shallow-copies the matrix and parameters. Creates a clone of the matrix
     * if it is on the wrong executor.
     */
    Isai& operator=(const Isai& other);

    /**
     * Move-assigns an ISAI preconditioner. Preserves the executor,
     * moves the matrix and parameters. Creates a clone of the matrix
     * if it is on the wrong executor. The moved-from object is empty (0x0
     * with nullptr matrix and default parameters)
     */
    Isai& operator=(Isai&& other);

    /**
     * Copy-constructs an ISAI preconditioner. Inherits the executor,
     * shallow-copies the matrix and parameters.
     */
    Isai(const Isai& other);

    /**
     * Move-constructs an ISAI preconditioner. Inherits the executor,
     * moves the matrix and parameters. The moved-from object is empty (0x0
     * with nullptr matrix and default parameters)
     */
    Isai(Isai&& other);

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

        /**
         * @brief Size limit for the excess system.
         *
         * For rows with more than 32 nonzero entries, the algorithm builds up
         * an excess system which is solved with sparse triangular solves (for
         * upper or lower ISAI) or GMRES (for general ISAI). If this parameter
         * is set to some m > 0, the excess system is solved as soon
         * as its size supersedes m. This is repeated until the complete excess
         * solution has been computed.
         * Must be at least 0, default value 0.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(excess_limit, 0u);

        /**
         * @brief Factory for the Excess System solver.
         *
         * Defaults to using a triangular solver for upper and lower ISAI and
         * to Block-Jacobi preconditioned GMRES for general and spd ISAI.
         */
        std::shared_ptr<LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            excess_solver_factory, nullptr);

        remove_complex<value_type> GKO_FACTORY_PARAMETER_SCALAR(
            excess_solver_reduction,
            static_cast<remove_complex<value_type>>(1e-6));
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
     * @param factory  the factory to use to create the preconditioner
     * @param system_matrix  the matrix for which an ISAI is to be computed
     */
    explicit Isai(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Isai>(factory->get_executor(), system_matrix->get_size()),
          parameters_{factory->get_parameters()}
    {
        const auto skip_sorting = parameters_.skip_sorting;
        const auto power = parameters_.sparsity_power;
        const auto excess_limit = parameters_.excess_limit;
        generate_inverse(system_matrix, skip_sorting, power, excess_limit,
                         static_cast<remove_complex<value_type>>(
                             parameters_.excess_solver_reduction));
        if (IsaiType == isai_type::spd) {
            auto inv = share(as<Csr>(approximate_inverse_));
            auto inv_transp = share(inv->conj_transpose());
            approximate_inverse_ =
                Composition<ValueType>::create(inv_transp, inv);
        }
    }

    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        approximate_inverse_->apply(b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
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
                          bool skip_sorting, int power, index_type excess_limit,
                          remove_complex<value_type> excess_solver_reduction);

private:
    std::shared_ptr<LinOp> approximate_inverse_;
};


template <typename ValueType = default_precision, typename IndexType = int32>
using LowerIsai = Isai<isai_type::lower, ValueType, IndexType>;

template <typename ValueType = default_precision, typename IndexType = int32>
using UpperIsai = Isai<isai_type::upper, ValueType, IndexType>;

template <typename ValueType = default_precision, typename IndexType = int32>
using GeneralIsai = Isai<isai_type::general, ValueType, IndexType>;

template <typename ValueType = default_precision, typename IndexType = int32>
using SpdIsai = Isai<isai_type::spd, ValueType, IndexType>;


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_ISAI_HPP_

// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_PAR_IC_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_PAR_IC_HPP_


#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


/**
 * ParIC is an incomplete Cholesky factorization which is computed in parallel.
 *
 * $L$ is a lower triangular matrix, which approximates a given matrix $A$ with
 * $A \approx LL^H$. Here, $L + L^H$ has the same sparsity pattern as $A$, which
 * is also called IC(0).
 *
 * The ParIC algorithm generates the incomplete factors iteratively, using a
 * fixed-point iteration of the form
 *
 * $
 * F(L) =
 * \begin{cases}
 *     \sqrt{a_{ii}-\sum_{k=1}^{i-1}|l_{ik}|^2}, \quad & i == j \\
 *     a_{ij}-\sum_{k=1}^{i-1}l_{ik}u_{kj}, \quad & i < j
 * \end{cases}
 * $
 *
 * In general, the entries of $L$ can be iterated in parallel and in
 * asynchronous fashion, the algorithm asymptotically converges to the
 * incomplete factors $L$ and $L^H$ fulfilling $\left(R = A - L \cdot
 * L^H\right)\vert_\mathcal{S} = 0\vert_\mathcal{S}$ where $\mathcal{S}$ is the
 * pre-defined sparsity pattern (in case of IC(0) the sparsity pattern of the
 * system matrix $A$). The number of ParIC sweeps needed for convergence
 * depends on the parallelism level: For sequential execution, a single sweep
 * is sufficient, for fine-grained parallelism, the number of sweeps necessary
 * to get a good approximation of the incomplete factors depends heavily on the
 * problem. On the OpenMP executor, 3 sweeps usually give a decent approximation
 * in our experiments, while GPU executors can take 10 or more iterations.
 *
 * The ParIC algorithm in Ginkgo follows the design of E. Chow and A. Patel,
 * Fine-grained Parallel Incomplete LU Factorization, SIAM Journal on Scientific
 * Computing, 37, C169-C193 (2015).
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ParIc : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;

    std::shared_ptr<const matrix_type> get_l_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
            this->get_operators()[0]);
    }

    std::shared_ptr<const matrix_type> get_lt_factor() const
    {
        if (this->get_operators().size() == 2) {
            // Can be `static_cast` since the type is guaranteed in this class
            return std::static_pointer_cast<const matrix_type>(
                this->get_operators()[1]);
        } else {
            return std::static_pointer_cast<const matrix_type>(
                share(get_l_factor()->conj_transpose()));
        }
    }

    // Remove the possibility of calling `create`, which was enabled by
    // `Composition`
    template <typename... Args>
    static std::unique_ptr<Composition<ValueType>> create(Args&&... args) =
        delete;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The number of iterations the `compute` kernel will use when doing
         * the factorization. The default value `0` means `Auto`, so the
         * implementation decides on the actual value depending on the
         * resources that are available.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(iterations, 0);

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, the factorization might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        /**
         * Strategy which will be used by the L matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(l_strategy, nullptr);

        /**
         * `true` will generate both L and L^H, `false` will only generate the L
         * factor, resulting in a Composition of only a single LinOp. This can
         * be used to avoid the transposition operation.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(both_factors, true);
    };
    GKO_ENABLE_LIN_OP_FACTORY(ParIc, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit ParIc(const Factory* factory,
                   std::shared_ptr<const LinOp> system_matrix)
        : Composition<ValueType>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        if (parameters_.l_strategy == nullptr) {
            parameters_.l_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        generate(system_matrix, parameters_.skip_sorting,
                 parameters_.both_factors)
            ->move_to(this);
    }

    std::unique_ptr<Composition<ValueType>> generate(
        const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
        bool both_factors) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_FACTORIZATION_PAR_IC_HPP_

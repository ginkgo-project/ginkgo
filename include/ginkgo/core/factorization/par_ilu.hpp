// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_PAR_ILU_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_PAR_ILU_HPP_


#include <memory>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


/**
 * ParILU is an incomplete LU factorization which is computed in parallel.
 *
 * \f$L\f$ is a lower unitriangular, while \f$U\f$ is an upper triangular
 * matrix, which approximate a given matrix \f$A\f$ with \f$A \approx LU\f$.
 * Here, \f$L\f$ and \f$U\f$ have the same sparsity pattern as \f$A\f$, which is
 * also called ILU(0).
 *
 * The ParILU algorithm generates the incomplete factors iteratively, using a
 * fixed-point iteration of the form
 *
 * \f[
 *   F(L, U)_{ij} = \begin{cases}
 *     \frac{1}{u_{jj}}
 *       \left( a_{ij} - \sum_{k=1}^{j-1} l_{ik} u_{kj} \right),
 *       & i > j, \\
 *     a_{ij} - \sum_{k=1}^{i-1} l_{ik} u_{kj},
 *       & i \leq j.
 *   \end{cases}
 * \f]
 *
 * In general, the entries of \f$L\f$ and \f$U\f$ can be iterated in parallel
 * and in asynchronous fashion; the algorithm asymptotically converges to
 * incomplete factors \f$L\f$ and \f$U\f$ fulfilling
 * \f$ (R = A - L U)\vert_\mathcal{S} = 0\vert_\mathcal{S} \f$
 * where \f$\mathcal{S}\f$ is the pre-defined sparsity pattern (in case of
 * ILU(0), the sparsity pattern of the system matrix \f$A\f$). The number of
 * ParILU sweeps needed for convergence depends on the parallelism level: for
 * sequential execution, a single sweep is sufficient; for fine-grained
 * parallelism, the number of sweeps necessary to get a good approximation
 * of the incomplete factors depends heavily on the problem. On the OpenMP
 * executor, 3 sweeps usually give a decent approximation in our experiments,
 * while GPU executors can take 10 or more iterations.
 *
 * @par References
 * - Chow, E., Patel, A.
 *   *Fine-Grained Parallel Incomplete LU Factorization.*
 *   SIAM Journal on Scientific Computing, 37 (2), C169–C193, 2015.
 *   <https://doi.org/10.1137/140968896>
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ParIlu : public Composition<ValueType> {
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using l_matrix_type = matrix_type;
    using u_matrix_type = matrix_type;

    std::shared_ptr<const matrix_type> get_l_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
            this->get_operators()[0]);
    }

    std::shared_ptr<const matrix_type> get_u_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
            this->get_operators()[1]);
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

        GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS

        /**
         * Strategy which will be used by the L matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        GKO_DEPRECATED("use matrix::csr::spmv_strategy instead")
        parameters_type& with_l_strategy(
            std::shared_ptr<typename matrix_type::strategy_type> value)
        {
            if (value) {
                this->l_strategy = value->get_enum();
            } else {
                this->l_strategy = matrix::csr::spmv_strategy::classical;
            }
            return *this;
        }

        GKO_END_DISABLE_DEPRECATION_WARNINGS

        /**
         * Strategy which will be used by the L matrix. The default value is
         * `classical`.
         */
        parameters_type& with_l_strategy(matrix::csr::spmv_strategy value)
        {
            this->l_strategy = value;
            return *this;
        }

        matrix::csr::spmv_strategy l_strategy{
            matrix::csr::spmv_strategy::classical};

        GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS

        /**
         * Strategy which will be used by the U matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        GKO_DEPRECATED("use matrix::csr::spmv_strategy instead")
        parameters_type& with_u_strategy(
            std::shared_ptr<typename matrix_type::strategy_type> value)
        {
            if (value) {
                this->u_strategy = value->get_enum();
            } else {
                this->u_strategy = matrix::csr::spmv_strategy::classical;
            }
            return *this;
        }

        GKO_END_DISABLE_DEPRECATION_WARNINGS

        /**
         * Strategy which will be used by the U matrix. The default value is
         * `classical`.
         */
        parameters_type& with_u_strategy(matrix::csr::spmv_strategy value)
        {
            this->u_strategy = value;
            return *this;
        }

        matrix::csr::spmv_strategy u_strategy{
            matrix::csr::spmv_strategy::classical};
    };
    GKO_ENABLE_LIN_OP_FACTORY(ParIlu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Create the parameters from the property_tree.
     * Because this is directly tied to the specific type, the value/index type
     * settings within config are ignored and type_descriptor is only used
     * for children configs.
     *
     * @param config  the property tree for setting
     * @param context  the registry
     * @param td_for_child  the type descriptor for children configs. The
     *                      default uses the value/index type of this class.
     *
     * @return parameters
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, IndexType>());

protected:
    explicit ParIlu(const Factory* factory,
                    std::shared_ptr<const LinOp> system_matrix)
        : Composition<ValueType>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        auto comp =
            generate_l_u(system_matrix, parameters_.skip_sorting,
                         parameters_.l_strategy, parameters_.u_strategy);
        for (auto& op : comp->get_operators()) {
            this->add_operators(op);
        }
    }

    /**
     * Generates the incomplete LU factors, which will be returned as a
     * composition of the lower (first element of the composition) and the
     * upper factor (second element). The dynamic type of L is l_matrix_type,
     * while the dynamic type of U is u_matrix_type.
     *
     * @param system_matrix  the source matrix used to generate the factors.
     *                       @note: system_matrix must be convertible to a Csr
     *                              Matrix, otherwise, an exception is thrown.
     * @param skip_sorting  if set to `true`, the sorting will be skipped.
     *                      @note: If the matrix is not sorted, the
     *                             factorization fails.
     * @param l_strategy  Strategy, which will be used by the L matrix.
     * @param u_strategy  Strategy, which will be used by the U matrix.
     * @return  A Composition, containing the incomplete LU factors for the
     *          given system_matrix (first element is L, then U)
     */
    std::unique_ptr<Composition<ValueType>> generate_l_u(
        const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
        matrix::csr::spmv_strategy l_strategy,
        matrix::csr::spmv_strategy u_strategy) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_FACTORIZATION_PAR_ILU_HPP_

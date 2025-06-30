// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_ILU_AT_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_ILU_AT_HPP_


#include <memory>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/incomplete_factorization.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


/**
 * Represents an incomplete LU factorization -- ILU -- of a sparse matrix, with
 * "automatic threshold" dropping.
 *
 * More specifically, it consists of a lower unitriangular factor $L$ and
 * an upper triangular factor $U$ with sparsity pattern $\mathcal S$ fulfilling
 * $LU = A$ on $\mathcal S$. The factorization allows for fill-in beginning with
 * "superlevels". For superlevel $k$, the initial sparsity pattern will include
 * all elements from the traditional level $k$, plus some additional fill-in.
 * This is done to perform a simpler (in terms of handling GPU memory) series of
 * operations for finding the fill-in, compared to the traditional computation
 * of fill-in levels. After the fatorization is performed on the superlevel
 * sparsity pattern, a threshold is dynamically selected as in ParILUT, and
 * locations below the threshold are dropped from the final factors.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class IluAt : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using sparsity_matrix_type = matrix::Csr<float, IndexType>;
    using SparsityType = float;

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
         * Strategy which will be used by the L matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(l_strategy, nullptr);

        /**
         * Strategy which will be used by the U matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(u_strategy, nullptr);

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, this factorization might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        /**
         * @brief the amount of fill-in that is allowed in L and U compared to
         *        the ILU(0) factorization.
         *
         * The threshold for removing candidates from the intermediate L and U
         * is set such that the resulting sparsity pattern has at most
         * `fill_in_limit` times the number of non-zeros of the ILU(0)
         * factorization.
         *
         * The default value `2.0` allows twice the number of non-zeros in
         * L and U compared to ILU(0).
         */
        double GKO_FACTORY_PARAMETER_SCALAR(fill_in_limit, 2.0);

        /**
         * Set the "superlevel" for the initial sparsity pattern (before
         * dropping).
         */
        int32 GKO_FACTORY_PARAMETER_SCALAR(superlevel, 0);
    };
    GKO_ENABLE_LIN_OP_FACTORY(IluAt, parameters, Factory);
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
    IluAt(const Factory* factory,
          std::shared_ptr<const gko::LinOp> system_matrix)
        : Composition<ValueType>{factory->get_executor()},
          parameters_{factory->get_parameters()}
    {
        if (parameters_.l_strategy == nullptr) {
            parameters_.l_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        if (parameters_.u_strategy == nullptr) {
            parameters_.u_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        generate_l_u(system_matrix, parameters_.skip_sorting)->move_to(this);
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
     * @param skip_sorting  determines if the sorting of system_matrix can be
     *                      skipped (therefore, marking that it is already
     *                      sorted)
     * @return  A Composition, containing the incomplete LU factors for the
     *          given system_matrix (first element is L, then U)
     */
    std::unique_ptr<Composition<ValueType>> generate_l_u(
        const std::shared_ptr<const LinOp>& system_matrix,
        bool skip_sorting) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_FACTORIZATION_ILU_AT_HPP_

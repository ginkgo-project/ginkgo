// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/factorization.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace experimental {
namespace factorization {


enum class symbolic_type {
    /** An LU factorization algorithm that works on all matrices. */
    general,
    /**
     * An LU factorization algorithm that works best on matrices with an almost
     * symmetric sparsity pattern. It is correct for general matrices, but may
     * use excessive amounts of memory and time.
     */
    near_symmetric,
    /**
     * An LU factorization algorithm that works only on matrices with a
     * symmetric sparsity pattern. Running it on a matrix with a non-symmetric
     * sparsity pattern is undefined behavior and will likely lead to the
     * application crashing.
     */
    symmetric
};


/**
 * Computes an LU factorization of a sparse matrix. This LinOpFactory returns a
 * Factorization storing the L and U factors for the provided system matrix in
 * matrix::Csr format. If no symbolic factorization is provided, it will be
 * computed first. It expects all fill-in entries to be present in the symbolic
 * factorization. If the symbolic factorization is missing some entries, please
 * refer to Ilu.
 *
 * @tparam ValueType  the type used to store values of the system matrix
 * @tparam IndexType  the type used to store sparsity pattern indices of the
 *                    system matrix
 *
 * @ingroup factor
 */
template <typename ValueType, typename IndexType>
class Lu
    : public EnablePolymorphicObject<Lu<ValueType, IndexType>, LinOpFactory>,
      public EnablePolymorphicAssignment<Lu<ValueType, IndexType>> {
public:
    struct parameters_type;
    friend class EnablePolymorphicObject<Lu, LinOpFactory>;
    friend class enable_parameters_type<parameters_type, Lu>;

    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<value_type, index_type>;
    using sparsity_pattern_type = matrix::SparsityCsr<value_type, index_type>;
    using factorization_type = Factorization<value_type, index_type>;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Lu> {
        /**
         * The combined sparsity pattern L + U of the factors L and U. It can be
         * used to avoid the potentially costly symbolic factorization of the
         * system matrix if its symbolic factorization is already known.
         * If it is set to nullptr, the symbolic factorization will be computed.
         * @note Currently, the symbolic factorization needs to be provided if
         *       the system matrix does not have a symmetric sparsity pattern.
         */
        std::shared_ptr<const sparsity_pattern_type>
            GKO_FACTORY_PARAMETER_SCALAR(symbolic_factorization, nullptr);

        /**
         * If the symbolic factorization of the matrix is not provided to the
         * factory, this parameter controls which algorithm will be used to
         * compute it.
         * @note Only use symbolic_factorization_algorithm::symmetric if you are
         *       sure your matrix has a symmetric sparsity pattern!
         */
        symbolic_type GKO_FACTORY_PARAMETER_SCALAR(symbolic_algorithm,
                                                   symbolic_type::general);

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, the algorithm may produce
         * incorrect results or crash.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };

    /**
     * Returns the parameters used to construct the factory.
     *
     * @return the parameters used to construct the factory.
     */
    const parameters_type& get_parameters() { return parameters_; }

    /**
     * @copydoc get_parameters
     */
    const parameters_type& get_parameters() const { return parameters_; }

    /**
     * @copydoc LinOpFactory::generate
     * @note This function overrides the default LinOpFactory::generate to
     *       return a Factorization instead of a generic LinOp, which would need
     *       to be cast to Factorization again to access its factors.
     *       It is only necessary because smart pointers aren't covariant.
     */
    std::unique_ptr<factorization_type> generate(
        std::shared_ptr<const LinOp> system_matrix) const;

    /** Creates a new parameter_type to set up the factory. */
    static parameters_type build() { return {}; }

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
    explicit Lu(std::shared_ptr<const Executor> exec,
                const parameters_type& params = {});

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system_matrix) const override;

private:
    parameters_type parameters_;
};


}  // namespace factorization
}  // namespace experimental
}  // namespace gko

// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_SOR_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_SOR_HPP_


#include <vector>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/config/config.hpp>


namespace gko {
namespace preconditioner {


/**
 * This class generates the (S)SOR preconditioner.
 *
 * The SOR preconditioner starts from a splitting of the the matrix $A$ into
 * $A = D + L + U$, where $L$ contains all entries below the diagonal, and $U$
 * contains all entries above the diagonal. The application of the
 * preconditioner is then defined as solving $M x = y$ with
 * \f[
 * M = \frac{1}{\omega} (D + \omega L), \quad 0 < \omega < 2.
 * \f]
 * $\omega$ is known as the relaxation factor.
 * The preconditioner can be made symmetric, leading to the SSOR preconitioner.
 * Here, $M$ is defined as
 * \f[
 * M = \frac{1}{\omega (2 - \omega)} (D + \omega L) D^{-1} (D + \omega U) ,
 * \quad 0 < \omega < 2.
 * \f]
 * A detailed description can be found in Iterative Methods for Sparse Linear
 * Systems (Y. Saad) ch. 4.1.
 *
 * This class is a factory, which will only generate the preconditioner. The
 * resulting LinOp will represent the application of $M^{-1}$.
 *
 * @tparam ValueType  The value type of the internally used CSR matrix
 * @tparam IndexType  The index type of the internally used CSR matrix
 *
 * @ingroup precond
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Sor
    : public EnablePolymorphicObject<Sor<ValueType, IndexType>, LinOpFactory>,
      public EnablePolymorphicAssignment<Sor<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Sor, LinOpFactory>;

public:
    struct parameters_type;
    friend class enable_parameters_type<parameters_type, Sor>;

    using value_type = ValueType;
    using index_type = IndexType;
    using composition_type = Composition<ValueType>;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Sor> {
        // skip sorting of input matrix
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        // determines if SOR or SSOR should be used
        bool GKO_FACTORY_PARAMETER_SCALAR(symmetric, false);

        // has to be between 0.0 and 2.0
        remove_complex<value_type> GKO_FACTORY_PARAMETER_SCALAR(
            relaxation_factor, remove_complex<value_type>(1.2));

        // factory for the lower triangular factor solver, defaults to LowerTrs
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            l_solver);

        // factory for the upper triangular factor solver, unused if symmetric
        // is false, defaults to UpperTrs
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            u_solver);
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
    std::unique_ptr<composition_type> generate(
        std::shared_ptr<const LinOp> system_matrix) const;

    /** Creates a new parameter_type to set up the factory. */
    static parameters_type build() { return {}; }

    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, IndexType>());

protected:
    explicit Sor(std::shared_ptr<const Executor> exec,
                 const parameters_type& params = {})
        : EnablePolymorphicObject<Sor, LinOpFactory>(exec), parameters_(params)
    {
        GKO_ASSERT(parameters_.relaxation_factor > 0.0 &&
                   parameters_.relaxation_factor < 2.0);
    }

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system_matrix) const override;

private:
    parameters_type parameters_;
};
}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_SOR_HPP_

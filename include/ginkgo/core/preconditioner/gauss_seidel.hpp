// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_GAUSS_SEIDEL_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_GAUSS_SEIDEL_HPP_


#include <vector>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/config/config.hpp>


namespace gko {
namespace preconditioner {


/**
 * This class generates the Gauss-Seidel preconditioner.
 *
 * This is the special case of the relaxation factor $\omega = 1$ of the (S)SOR
 * preconditioner.
 *
 * @see Sor
 *
 * @ingroup precond
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class GaussSeidel
    : public EnablePolymorphicObject<GaussSeidel<ValueType, IndexType>,
                                     LinOpFactory>,
      public EnablePolymorphicAssignment<GaussSeidel<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<GaussSeidel, LinOpFactory>;

public:
    struct parameters_type;
    friend class enable_parameters_type<parameters_type, GaussSeidel>;

    using value_type = ValueType;
    using index_type = IndexType;
    using composition_type = Composition<ValueType>;

    struct parameters_type
        : public enable_parameters_type<parameters_type, GaussSeidel> {
        // skip sorting of input matrix
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        // determines if Gauss-Seidel or symmetric Gauss-Seidel should be used
        bool GKO_FACTORY_PARAMETER_SCALAR(symmetric, false);

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
    explicit GaussSeidel(std::shared_ptr<const Executor> exec,
                         const parameters_type& params = {})
        : EnablePolymorphicObject<GaussSeidel, LinOpFactory>(exec),
          parameters_(params)
    {}

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system_matrix) const override;

private:
    parameters_type parameters_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_GAUSS_SEIDEL_HPP_

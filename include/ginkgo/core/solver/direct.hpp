// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_DIRECT_HPP_
#define GKO_PUBLIC_CORE_SOLVER_DIRECT_HPP_


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/factorization/factorization.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/solver/triangular.hpp>


namespace gko {
namespace experimental {
namespace solver {


/**
 * A direct solver based on a factorization into lower and upper triangular
 * factors (with an optional diagonal scaling).
 * The solver is built from the Factorization returned by the provided
 * LinOpFactory.
 *
 * @tparam ValueType  the type used to store values of the system matrix
 * @tparam IndexType  the type used to store sparsity pattern indices of the
 *                    system matrix
 */
template <typename ValueType, typename IndexType>
class Direct : public EnableLinOp<Direct<ValueType, IndexType>>,
               public gko::solver::EnableSolverBase<
                   Direct<ValueType, IndexType>,
                   factorization::Factorization<ValueType, IndexType>>,
               public Transposable {
    friend class EnablePolymorphicObject<Direct, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using factorization_type =
        factorization::Factorization<value_type, index_type>;
    using transposed_type = Direct;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    class Factory;

    struct parameters_type : enable_parameters_type<parameters_type, Factory> {
        /**
         * Number of right hand sides.
         *
         * @note This value is currently only required for the CUDA executor,
         *       which will throw an exception if a different number of rhs is
         *       passed to Direct::apply.
         */
        gko::size_type GKO_FACTORY_PARAMETER_SCALAR(num_rhs, 1u);

        /** The factorization factory to use for generating the factors. */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            factorization);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Direct, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /** Creates a copy of the solver. */
    Direct(const Direct&);

    /** Moves from the given solver, leaving it empty. */
    Direct(Direct&&);

    Direct& operator=(const Direct&);

    Direct& operator=(Direct&&);

protected:
    explicit Direct(std::shared_ptr<const Executor> exec);

    Direct(const Factory* factory, std::shared_ptr<const LinOp> system_matrix);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    using lower_type = gko::solver::LowerTrs<value_type, index_type>;
    using upper_type = gko::solver::UpperTrs<value_type, index_type>;

    std::unique_ptr<lower_type> lower_solver_;
    std::unique_ptr<upper_type> upper_solver_;
};


}  // namespace solver
}  // namespace experimental


namespace solver {


template <typename ValueType, typename IndexType>
struct workspace_traits<
    gko::experimental::solver::Direct<ValueType, IndexType>> {
    using Solver = gko::experimental::solver::Direct<ValueType, IndexType>;
    // number of vectors used by this workspace
    static int num_vectors(const Solver&);
    // number of arrays used by this workspace
    static int num_arrays(const Solver&);
    // array containing the num_vectors names for the workspace vectors
    static std::vector<std::string> op_names(const Solver&);
    // array containing the num_arrays names for the workspace vectors
    static std::vector<std::string> array_names(const Solver&);
    // array containing all varying scalar vectors (independent of problem size)
    static std::vector<int> scalars(const Solver&);
    // array containing all varying vectors (dependent on problem size)
    static std::vector<int> vectors(const Solver&);

    // intermediate vector
    constexpr static int intermediate = 0;
};


}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_DIRECT_HPP_

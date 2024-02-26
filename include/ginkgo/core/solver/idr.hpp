// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_IDR_HPP_
#define GKO_PUBLIC_CORE_SOLVER_IDR_HPP_


#include <random>
#include <typeinfo>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
/**
 * @brief The ginkgo Solver namespace.
 *
 * @ingroup solvers
 */
namespace solver {


/**
 * IDR(s) is an efficient method for solving large nonsymmetric systems of
 * linear equations. The implemented version is the one presented in the
 * paper "Algorithm 913: An elegant IDR(s) variant that efficiently exploits
 * biorthogonality properties" by M. B. Van Gijzen and P. Sonneveld.
 *
 * The method is based on the induced dimension reduction theorem which
 * provides a way to construct subsequent residuals that lie in a sequence
 * of shrinking subspaces. These subspaces are spanned by s vectors which are
 * first generated randomly and then orthonormalized. They are stored in
 * a dense matrix.
 *
 * @tparam ValueType  precision of the elements of the system matrix.
 *
 * @ingroup idr
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Idr
    : public EnableLinOp<Idr<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType, Idr<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Idr>;
    friend class EnablePolymorphicObject<Idr, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = Idr<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    /**
     * Gets the subspace dimension of the solver.
     *
     * @return the subspace Dimension*/
    size_type get_subspace_dim() const { return parameters_.subspace_dim; }

    /**
     * Sets the subspace dimension of the solver.
     *
     * @param other  the new subspace Dimension*/
    void set_subspace_dim(const size_type other)
    {
        parameters_.subspace_dim = other;
    }

    /**
     * Gets the kappa parameter of the solver.
     *
     * @return the kappa parameter
     */
    remove_complex<ValueType> get_kappa() const { return parameters_.kappa; }

    /**
     * Sets the kappa parameter of the solver.
     *
     * @param other  the new kappa parameter
     */
    void set_kappa(const remove_complex<ValueType> other)
    {
        parameters_.kappa = other;
    }

    /**
     * Gets the deterministic parameter of the solver.
     *
     * @return the deterministic parameter
     */
    bool get_deterministic() const { return parameters_.deterministic; }

    /**
     * Sets the deterministic parameter of the solver.
     *
     * @param other  the new deterministic parameter
     */
    void set_deterministic(const bool other)
    {
        parameters_.deterministic = other;
    }

    /**
     * Gets the complex_subspace parameter of the solver.
     *
     * @return the complex_subspace parameter
     */
    bool get_complex_subspace() const { return parameters_.complex_subspace; }

    /**
     * Sets the complex_subspace parameter of the solver.
     *
     * @param other  the new complex_subspace parameter
     * @deprecated Please use set_complex_subspace instead
     */
    GKO_DEPRECATED("Use set_complex_subspace instead")
    void set_complex_subpsace(const bool other)
    {
        this->set_complex_subspace(other);
    }

    /**
     * Sets the complex_subspace parameter of the solver.
     *
     * @param other  the new complex_subspace parameter
     */
    void set_complex_subspace(const bool other)
    {
        parameters_.complex_subspace = other;
    }

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {
        /**
         * Dimension of the subspace S. Determines how many intermediate
         * residuals are computed in each iteration.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(subspace_dim, 2u);

        /**
         * Threshold to determine if Av_n and v_n are too close to being
         * perpendicular.
         * This is considered to be the case if
         * $|(Av_n)^H * v_n / (norm(Av_n) * norm(v_n))| < kappa$
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(kappa, 0.7);

        /**
         * If set to true, the vectors spanning the subspace S are chosen
         * deterministically. This is mostly needed for testing purposes.
         *
         * Note: If 'deterministic' is set to true, the subspace vectors are
         * generated in serial on the CPU, which can be very slow.
         *
         * The default behaviour is to choose the subspace vectors randomly.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(deterministic, false);

        /**
         * If set to true, IDR will use a complex subspace S also for real
         * problems, allowing for faster convergence and better results by
         * acknowledging the influence of complex eigenvectors.
         *
         * The default is false.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(complex_subspace, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Idr, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    template <typename VectorType>
    void iterate(const VectorType* dense_b, VectorType* dense_x) const;

    explicit Idr(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Idr>(std::move(exec))
    {}

    explicit Idr(const Factory* factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Idr>(factory->get_executor(),
                           gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Idr<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<Idr<ValueType>> {
    using Solver = Idr<ValueType>;
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

    // residual vector
    constexpr static int residual = 0;
    // v vector
    constexpr static int v = 1;
    // t vector
    constexpr static int t = 2;
    // helper vector
    constexpr static int helper = 3;
    // m multivector
    constexpr static int m = 4;
    // g multivector
    constexpr static int g = 5;
    // u multivector
    constexpr static int u = 6;
    // subspace multivector
    constexpr static int subspace = 7;
    // f "multiscalar"
    constexpr static int f = 8;
    // c "multiscalar"
    constexpr static int c = 9;
    // omega scalar
    constexpr static int omega = 10;
    // residual norm scalar
    constexpr static int residual_norm = 11;
    // T^H*T scalar
    constexpr static int tht = 12;
    // alpha "multiscalar"
    constexpr static int alpha = 13;
    // constant 1.0 scalar
    constexpr static int one = 14;
    // constant -1.0 scalar
    constexpr static int minus_one = 15;
    // constant -1.0 scalar
    constexpr static int subspace_minus_one = 16;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_IDR_HPP_

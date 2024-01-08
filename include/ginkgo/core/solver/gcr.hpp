// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_GCR_HPP_
#define GKO_PUBLIC_CORE_SOLVER_GCR_HPP_


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
namespace solver {


constexpr size_type gcr_default_krylov_dim = 100u;


/**
 * GCR or the generalized conjugate residual method is an iterative type Krylov
 * subspace method similar to GMRES which is suitable for nonsymmetric linear
 * systems.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of GCR are
 * merged into one step. Modified Gram-Schmidt is used.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Gcr
    : public EnableLinOp<Gcr<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType, Gcr<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Gcr>;
    friend class EnablePolymorphicObject<Gcr, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = Gcr<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    /**
     * Gets the Krylov dimension of the solver
     *
     * @return the Krylov dimension
     */
    size_type get_krylov_dim() const { return parameters_.krylov_dim; }

    /**
     * Sets the Krylov dimension
     *
     * @param other  the new Krylov dimension
     */
    void set_krylov_dim(size_type other) { parameters_.krylov_dim = other; }

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {
        /** Krylov subspace dimension/restart value. */
        size_type GKO_FACTORY_PARAMETER_SCALAR(krylov_dim, 0u);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Gcr, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Gcr(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Gcr>(std::move(exec))
    {}

    explicit Gcr(const Factory* factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Gcr>(factory->get_executor(),
                           gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Gcr<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {
        if (!parameters_.krylov_dim) {
            parameters_.krylov_dim = gcr_default_krylov_dim;
        }
    }
};


template <typename ValueType>
struct workspace_traits<Gcr<ValueType>> {
    using Solver = Gcr<ValueType>;
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
    // preconditioned vector
    constexpr static int precon_residual = 1;
    // A* preconditioned vector
    constexpr static int A_precon_residual = 2;
    // krylov bases (p in the algorithm)
    constexpr static int krylov_bases_p = 3;
    // mapped krylov bases (Ap in the algorithm)
    constexpr static int mapped_krylov_bases_Ap = 4;
    // tmp rAp parameter (r dot Ap in the algorithm)
    constexpr static int tmp_rAp = 5;
    // tmp minus beta parameter (-beta in the algorithm)
    constexpr static int tmp_minus_beta = 6;
    // array of norms of Ap
    constexpr static int Ap_norms = 7;
    // residual norm scalar
    constexpr static int residual_norm = 8;
    // constant 1.0 scalar
    constexpr static int one = 9;
    // constant -1.0 scalar
    constexpr static int minus_one = 10;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
    // final iteration number array
    constexpr static int final_iter_nums = 2;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_GCR_HPP_

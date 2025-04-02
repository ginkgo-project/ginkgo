// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_GMRES_HPP_
#define GKO_PUBLIC_CORE_SOLVER_GMRES_HPP_


#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


[[deprecated]] constexpr size_type default_krylov_dim = 100u;

constexpr size_type gmres_default_krylov_dim = 100u;

namespace gmres {
/**
 * Set the orthogonalization method for the Krylov subspace.
 */
enum class ortho_method {
    /**
     * Modified Gram-Schmidt (default)
     */
    mgs,
    /**
     * Classical Gram-Schmidt
     */
    cgs,
    /**
     * Classical Gram-Schmidt with re-orthogonalization
     */
    cgs2,
    /**
     * Randomized Gram-Schmidt
     */
    rgs
};

/** Prints an orthogonalization method. */
std::ostream& operator<<(std::ostream& stream, ortho_method ortho);

}  // namespace gmres

/**
 * GMRES or the generalized minimal residual method is an iterative type Krylov
 * subspace method which is suitable for nonsymmetric linear systems.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of GMRES are
 * merged into 2 separate steps. Modified Gram-Schmidt is used.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Gmres
    : public EnableLinOp<Gmres<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType, Gmres<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Gmres>;
    friend class EnablePolymorphicObject<Gmres, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = Gmres<ValueType>;

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

        /** Flexible GMRES */
        bool GKO_FACTORY_PARAMETER_SCALAR(flexible, false);

        /** Orthogonalization method */
        gmres::ortho_method GKO_FACTORY_PARAMETER_SCALAR(
            ortho_method, gmres::ortho_method::mgs);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Gmres, parameters, Factory);
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
     *                      default uses the value type of this class.
     *
     * @return parameters
     */
    static parameters_type parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child =
                                     config::make_type_descriptor<ValueType>());

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Gmres(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Gmres>(std::move(exec))
    {}

    explicit Gmres(const Factory* factory,
                   std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Gmres>(factory->get_executor(),
                             gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Gmres<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {
        if (!parameters_.krylov_dim) {
            parameters_.krylov_dim = gmres_default_krylov_dim;
        }
    }
};


template <typename ValueType>
struct workspace_traits<Gmres<ValueType>> {
    using Solver = Gmres<ValueType>;
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
    constexpr static int preconditioned_vector = 1;
    // krylov basis multivector
    constexpr static int krylov_bases = 2;
    // hessenberg matrix
    constexpr static int hessenberg = 3;
    // auxiliary space for CGS2
    constexpr static int hessenberg_aux = 4;
    // givens sin parameters
    constexpr static int givens_sin = 5;
    // givens cos parameters
    constexpr static int givens_cos = 6;
    // coefficients of the residual in Krylov space
    constexpr static int residual_norm_collection = 7;
    // residual norm scalar
    constexpr static int residual_norm = 8;
    // solution of the least-squares problem in Krylov space
    constexpr static int y = 9;
    // solution of the least-squares problem mapped to the full space
    constexpr static int before_preconditioner = 10;
    // preconditioned solution of the least-squares problem
    constexpr static int after_preconditioner = 11;
    // constant 1.0 scalar
    constexpr static int one = 12;
    // constant -1.0 scalar
    constexpr static int minus_one = 13;
    // temporary norm vector of next_krylov to copy into hessenberg matrix
    constexpr static int next_krylov_norm_tmp = 14;
    // preconditioned krylov basis multivector
    constexpr static int preconditioned_krylov_bases = 15;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
    // final iteration number array
    constexpr static int final_iter_nums = 2;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_GMRES_HPP_

// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_MULTIGRID_HPP_
#define GKO_PUBLIC_CORE_SOLVER_MULTIGRID_HPP_


#include <complex>
#include <functional>
#include <memory>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * @brief The solver multigrid namespace.
 *
 * @ingroup solver
 */
namespace multigrid {


/**
 * cycle defines which kind of multigrid cycle can be used.
 * It contains V, W, and F cycle.
 * - V, W cycle uses the algorithm according to Briggs, Henson, and McCormick: A
 *   multigrid tutorial 2nd Edition.
 * - F cycle uses the algorithm according to Trottenberg, Oosterlee, and
 *   Schuller: Multigrid 1st Edition. F cycle first uses the recursive call but
 *   second uses the V-cycle call such that F-cycle is between V and W cycle.
 */
enum class cycle { v, f, w };


/**
 * mid_smooth_type gives the options to handle the middle smoother
 * behavior between the two cycles in the same level. It only affects the
 * behavior when there's no operation between the post smoother of previous
 * cycle and the pre smoother of next cycle. Thus, it only affects W cycle and F
 * cycle.
 * - both: gives the same behavior as the original algorithm, which use posts
 *   smoother from previous cycle and pre smoother from next cycle.
 * - post_smoother: only uses the post smoother of previous cycle in the mid
 *   smoother
 * - pre_smoother: only uses the pre smoother of next cycle in the mid smoother
 * - standalone: uses the defined smoother in the mid smoother
 */
enum class mid_smooth_type { both, post_smoother, pre_smoother, standalone };


namespace detail {


// It should only be used internally
class MultigridState;


}  // namespace detail
}  // namespace multigrid


/**
 * Multigrid methods have a hierarchy of many levels, whose corase level is a
 * subset of the fine level, of the problem. The coarse level solves the system
 * on the residual of fine level and fine level will use the coarse solution to
 * correct its own result. Multigrid solves the problem by relatively cheap step
 * in each level and refining the result when prolongating back.
 *
 * The main step of each level
 * - Presmooth (solve on the fine level)
 * - Calculate residual
 * - Restrict (reduce the problem dimension)
 * - Solve residual in next level
 * - Prolongate (return to the fine level size)
 * - Postsmooth (correct the answer in fine level)
 *
 * Ginkgo uses the index from 0 for finest level (original problem size) ~ N for
 * the coarsest level (the coarsest solver), and its level counts is N (N
 * multigrid level generation).
 *
 * @ingroup Multigrid
 * @ingroup solvers
 * @ingroup LinOp
 */
class Multigrid : public EnableLinOp<Multigrid>,
                  public EnableSolverBase<Multigrid>,
                  public EnableIterativeBase<Multigrid>,
                  public EnableApplyWithInitialGuess<Multigrid> {
    friend class EnableLinOp<Multigrid>;
    friend class EnablePolymorphicObject<Multigrid, LinOp>;
    friend class EnableApplyWithInitialGuess<Multigrid>;

public:
    /**
     * Return true as iterative solvers use the data in x as an initial guess or
     * false if multigrid always set the input as zero
     *
     * @return bool  it is related to parameters variable zero_guess
     */
    bool apply_uses_initial_guess() const override
    {
        return this->get_default_initial_guess() ==
               initial_guess_mode::provided;
    }

    /**
     * Gets the list of MultigridLevel operators.
     *
     * @return the list of MultigridLevel operators
     */
    std::vector<std::shared_ptr<const gko::multigrid::MultigridLevel>>
    get_mg_level_list() const
    {
        return mg_level_list_;
    }

    /**
     * Gets the list of pre-smoother operators.
     *
     * @return the list of pre-smoother operators
     */
    std::vector<std::shared_ptr<const LinOp>> get_pre_smoother_list() const
    {
        return pre_smoother_list_;
    }

    /**
     * Gets the list of mid-smoother operators.
     *
     * @return the list of mid-smoother operators
     */
    std::vector<std::shared_ptr<const LinOp>> get_mid_smoother_list() const
    {
        return mid_smoother_list_;
    }

    /**
     * Gets the list of post-smoother operators.
     *
     * @return the list of post-smoother operators
     */
    std::vector<std::shared_ptr<const LinOp>> get_post_smoother_list() const
    {
        return post_smoother_list_;
    }

    /**
     * Gets the operator at the coarsest level.
     *
     * @return the coarsest operator
     */
    std::shared_ptr<const LinOp> get_coarsest_solver() const
    {
        return coarsest_solver_;
    }

    /**
     * Get the cycle of multigrid
     *
     * @return the multigrid::cycle
     */
    multigrid::cycle get_cycle() const { return parameters_.cycle; }

    /**
     * Set the cycle of multigrid
     *
     * @param multigrid::cycle the new cycle
     */
    void set_cycle(multigrid::cycle cycle) { parameters_.cycle = cycle; }


    class Factory;

    struct parameters_type
        : public enable_iterative_solver_factory_parameters<parameters_type,
                                                            Factory> {
        /**
         * MultigridLevel Factory list
         */
        std::vector<std::shared_ptr<const LinOpFactory>>
            GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(mg_level);

        /**
         * Custom selector size_type (size_type level, const LinOp* fine_matrix)
         * Selector function returns the element index in the vector for any
         * given level index and the matrix of the fine level.
         * For each level, this function is used to select the smoothers
         * and multigrid level generation from the respective lists.
         * For example,
         * ```
         * [](size_type level, const LinOp* fine_matrix) {
         *     if (level < 3) {
         *         return size_type{0};
         *     } else if (matrix->get_size()[0] > 1024) {
         *         return size_type{1};
         *     } else {
         *         return size_type{2};
         *     }
         * }
         * ```
         * It uses the 0-idx element if level < 3, the 1-idx element if level
         * >= 3 and the number of rows of fine matrix > 1024, or the 2-idx
         * elements otherwise.
         *
         * default selector:
         *     use the first factory when mg_level size = 1
         *     use the level as the index when mg_level size > 1
         */
        std::function<size_type(const size_type, const LinOp*)>
            GKO_FACTORY_PARAMETER_SCALAR(level_selector, nullptr);

        /**
         * Pre-smooth Factory list.
         * Its size must be 0, 1 or be the same as mg_level's.
         * when size = 0, use default smoother
         * when size = 1, use the first factory
         * when size > 1, use the same selector as mg_level
         *
         * If this option is not set (i.e. size = 0) then the default smoother
         * is used. The default smoother is one step of iterative refinement
         * with a scalar Jacobi preconditioner.
         *
         * If any element in the vector is a `nullptr` then the smoother
         * application at the corresponding level is skipped.
         */
        std::vector<std::shared_ptr<const LinOpFactory>>
            GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(pre_smoother);

        /**
         * Post-smooth Factory list.
         * It is similar to Pre-smooth Factory list. It is ignored if
         * the factory parameter post_uses_pre is set to true.
         */
        std::vector<std::shared_ptr<const LinOpFactory>>
            GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(post_smoother);

        /**
         * Mid-smooth Factory list. If it contains available elements, multigrid
         * always generates the corresponding smoother. However, it is only
         * involved in the procedure when cycle is F or W. It is similar to
         * Pre-smooth Factory list. It is ignored if the factory parameter
         * mid_case is not mid.
         */
        std::vector<std::shared_ptr<const LinOpFactory>>
            GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(mid_smoother);

        /**
         * Whether post-smoothing-related calls use corresponding
         * pre-smoothing-related calls.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(post_uses_pre, true);

        /**
         * Choose the behavior of mid smoother between two cycles close to each
         * other in the same level. The default is
         * multigrid::mid_smooth_type::standalone.
         *
         * @see enum multigrid::mid_smooth_type
         */
        multigrid::mid_smooth_type GKO_FACTORY_PARAMETER_SCALAR(
            mid_case, multigrid::mid_smooth_type::standalone);

        /**
         * The maximum number of mg_level (without coarsest solver level) that
         * can be used.
         *
         * If the multigrid hit the max_levels limit, including the coarsest
         * solver level contains max_levels + 1 levels.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(max_levels, 10);

        /**
         * The minimal number of coarse rows.
         * If generation gets the matrix which contains less than
         * `min_coarse_rows`, the generation stops.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(min_coarse_rows, 64);

        /**
         * Coarsest factory list.
         *
         * If not set, then a direct LU solver will be used as solver on the
         * coarsest level.
         */
        std::vector<std::shared_ptr<const LinOpFactory>>
            GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(coarsest_solver);

        /**
         * Custom coarsest_solver selector
         * size_type (size_type level, const LinOp *coarsest_matrix)
         * Selector function returns the correct element index in the
         * vector for any given level index and the system matrix.
         * It can be used to choose the sparse iterative solver if the coarsest
         * matrix is still large but the dense solver if the matrix is small
         * enough. For example,
         * ```
         * [](size_type level, const LinOp* coarsest_matrix) {
         *     if (coarsest_matrix->get_size()[0] > 1024) {
         *         return size_type{0};
         *     } else {
         *         return size_type{1};
         *     }
         * }
         * ```
         * Coarsest solver uses the 0-idx element if the number of rows of the
         * coarsest_matrix > 1024 or 1-idx element for other cases.
         *
         * default selector: use the first factory
         */
        std::function<size_type(const size_type, const LinOp*)>
            GKO_FACTORY_PARAMETER_SCALAR(solver_selector, nullptr);

        /**
         * Multigrid cycle type. The default is multigrid::cycle::v.
         *
         * @see enum multigrid::cycle
         */
        multigrid::cycle GKO_FACTORY_PARAMETER_SCALAR(cycle,
                                                      multigrid::cycle::v);

        /**
         * kcycle_base is a factor to choose how often enable FCG/GCR step.
         * This parameter is ignored on v, w, f cycle.
         * Enable the FCG/GCR step when level % kcycle_base == 0 and the next
         * level is not coarsest level.
         *
         * @note the FCG/GCR step works on the vectors of next level.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(kcycle_base, 1);

        /**
         * kcycle_rel_tol decides whether run the second iteration of FCG/GCR
         * step.
         * kcycle_rel_tol <= 0: always run one iterations.
         * kcycle_rel_tol == nan: always run two iterations.
         * ||updated_r|| <= kcycle_rel_tol ||r||: run second iteration.
         */
        double GKO_FACTORY_PARAMETER_SCALAR(kcycle_rel_tol, 0.25);

        /**
         * smoother_relax is the relaxation factor of default generated
         * smoother.
         */
        std::complex<double> GKO_FACTORY_PARAMETER_SCALAR(smoother_relax, 0.9);

        /**
         * smoother_iters is the number of iteration of default generated
         * smoother.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(smoother_iters, 1);

        /**
         * Default initial guess mode. The available options are under
         * initial_guess_mode.
         */
        initial_guess_mode GKO_FACTORY_PARAMETER_SCALAR(
            default_initial_guess, initial_guess_mode::zero);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Multigrid, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void apply_with_initial_guess_impl(const LinOp* b, LinOp* x,
                                       initial_guess_mode guess) const override;

    void apply_with_initial_guess_impl(const LinOp* alpha, const LinOp* b,
                                       const LinOp* beta, LinOp* x,
                                       initial_guess_mode guess) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x,
                          initial_guess_mode guess) const;

    /**
     * Generates the analysis structure from the system matrix and the right
     * hand side needed for the level solver.
     */
    void generate();

    explicit Multigrid(std::shared_ptr<const Executor> exec);

    explicit Multigrid(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix);

    /**
     * validate checks the given parameters are valid or not.
     */
    void validate();

    /**
     * verify_legal_length is to check whether the given len is legal for
     * ref_len if checked is activated. Throw GKO_NOT_SUPPORTED if the length is
     * illegal.
     *
     * @param checked  whether check the length
     * @param len  the length of input
     * @param ref_len  the length of reference
     */
    void verify_legal_length(bool checked, size_type len, size_type ref_len);

    void create_state() const;

private:
    std::vector<std::shared_ptr<const gko::multigrid::MultigridLevel>>
        mg_level_list_{};
    std::vector<std::shared_ptr<const LinOp>> pre_smoother_list_{};
    std::vector<std::shared_ptr<const LinOp>> mid_smoother_list_{};
    std::vector<std::shared_ptr<const LinOp>> post_smoother_list_{};
    std::shared_ptr<const LinOp> coarsest_solver_{};
    std::function<size_type(const size_type, const LinOp*)> level_selector_;
    std::function<size_type(const size_type, const LinOp*)> solver_selector_;

    /**
     * Manages MultigridState as a cache, so there is no need to allocate them
     * every time an intermediate vector is required. Copying an instance
     * will only yield an empty object since copying the cached vector would
     * not make sense.
     *
     * @internal  The struct is present so the whole class can be copyable
     *            (could also be done with writing `operator=` and copy
     *            constructor of the enclosing class by hand)
     */
    mutable struct cache_struct {
        cache_struct() = default;

        ~cache_struct() = default;

        cache_struct(const cache_struct&) {}

        cache_struct(cache_struct&&) {}

        cache_struct& operator=(const cache_struct&) { return *this; }

        cache_struct& operator=(cache_struct&&) { return *this; }

        // unique_ptr with default destructor does not work with the incomplete
        // type.
        std::shared_ptr<multigrid::detail::MultigridState> state{};
    } cache_;
};

template <>
struct workspace_traits<Multigrid> {
    using Solver = Multigrid;
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

    // stopping status array
    constexpr static int stop = 0;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_MULTIGRID_HPP_

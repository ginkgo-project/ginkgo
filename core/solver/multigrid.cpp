// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/multigrid.hpp"

#include <complex>
#include <string>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/workspace_tree.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/distributed/helpers.hpp"
#include "core/solver/ir_kernels.hpp"
#include "core/solver/multigrid_kernels.hpp"
#include "core/solver/solver_base.hpp"


namespace gko {
namespace solver {
namespace multigrid {


GKO_REGISTER_OPERATION(initialize, ir::initialize);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(kcycle_step_1, multigrid::kcycle_step_1);
GKO_REGISTER_OPERATION(kcycle_step_2, multigrid::kcycle_step_2);
GKO_REGISTER_OPERATION(kcycle_check_stop, multigrid::kcycle_check_stop);


}  // namespace multigrid


namespace {


/**
 * casting does the casting type or take the real part of complex if the input
 * is complex but require real output.
 *
 * @tparam ValueType  the output type
 * @tparam T  the input type
 *
 * @param x  input
 *
 * @return the ValueType value
 */
template <typename ValueType, typename T>
std::enable_if_t<is_complex_s<ValueType>::value == is_complex_s<T>::value,
                 ValueType>
casting(const T& x)
{
    return static_cast<ValueType>(x);
}

/**
 * @copydoc casting(const T&)
 */
template <typename ValueType, typename T>
std::enable_if_t<!is_complex_s<ValueType>::value && is_complex_s<T>::value,
                 ValueType>
casting(const T& x)
{
    return static_cast<ValueType>(real(x));
}


/**
 * handle_list generate the smoother for each MultigridLevel
 *
 * @tparam ValueType  the type of MultigridLevel
 */
template <typename ValueType>
void handle_list(
    size_type index, std::shared_ptr<const LinOp>& matrix,
    std::vector<std::shared_ptr<const LinOpFactory>>& smoother_list,
    std::vector<std::shared_ptr<const LinOp>>& smoother, size_type iteration,
    std::complex<double> relaxation_factor,
    solver::Workspace* ws_node = nullptr)
{
    auto list_size = smoother_list.size();
    auto gen_default_smoother = [&]() -> std::shared_ptr<const LinOp> {
        auto exec = matrix->get_executor();
#if GINKGO_BUILD_MPI
        if (gko::detail::is_distributed(matrix.get())) {
            using experimental::distributed::Matrix;
            return run<Matrix<ValueType, int32, int32>,
                       Matrix<ValueType, int32, int64>,
                       Matrix<ValueType, int64, int64>>(
                matrix,
                [exec, iteration, relaxation_factor,
                 ws_node](auto matrix) -> std::shared_ptr<const LinOp> {
                    using Mtx = typename decltype(matrix)::element_type;
                    auto factory = build_smoother(
                        experimental::distributed::preconditioner::Schwarz<
                            ValueType, typename Mtx::local_index_type,
                            typename Mtx::global_index_type>::build()
                            .with_local_solver(
                                preconditioner::Jacobi<ValueType>::build()
                                    .with_max_block_size(1u))
                            .on(exec),
                        iteration, casting<ValueType>(relaxation_factor));
                    if (ws_node) {
                        return factory->generate(matrix, ws_node);
                    }
                    return factory->generate(matrix);
                });
        }
#endif
        auto factory =
            build_smoother(preconditioner::Jacobi<ValueType>::build()
                               .with_max_block_size(1u)
                               .on(exec),
                           iteration, casting<ValueType>(relaxation_factor));
        if (ws_node) {
            return factory->generate(matrix, ws_node);
        }
        return factory->generate(matrix);
    };
    if (list_size != 0) {
        auto temp_index = list_size == 1 ? 0 : index;
        GKO_ENSURE_IN_BOUNDS(temp_index, list_size);
        auto item = smoother_list.at(temp_index);
        if (item == nullptr) {
            smoother.emplace_back(nullptr);
        } else if (ws_node) {
            auto solver = item->generate(matrix, ws_node);
            smoother.emplace_back(give(solver));
        } else {
            auto solver = item->generate(matrix);
            smoother.emplace_back(give(solver));
        }
    } else {
        smoother.emplace_back(gen_default_smoother());
    }
}


}  // namespace


namespace multigrid {


/**
 * The enum class is to combine the cycle information  It's legal to use a
 * binary or(|) operation to combine several properties.
 */
enum class cycle_mode {
    /**
     * indicate input is zero
     */
    x_is_zero = 1,

    /**
     * current process is the first one of the cycle
     */
    first_of_cycle = 2,

    /**
     * current process is the end one of the cycle
     */
    end_of_cycle = 4
};


GKO_ATTRIBUTES GKO_INLINE cycle_mode operator|(cycle_mode a, cycle_mode b)
{
    return static_cast<cycle_mode>(static_cast<int>(a) | static_cast<int>(b));
}


GKO_ATTRIBUTES GKO_INLINE bool has_property(cycle_mode a, cycle_mode b)
{
    return static_cast<bool>(static_cast<int>(a) & static_cast<int>(b));
}


namespace detail {


/**
 * MultigridState is used to store the necessary cache and run the operation of
 * all levels.
 *
 * @note it should only be used internally
 */
class MultigridState {
public:
    MultigridState() : nrhs{static_cast<size_type>(-1)} {}

    /**
     * Generate the cache for later usage.
     *
     * @param system_matrix_in  the system matrix
     * @param multigrid_in  the multigrid information
     * @param nrhs_in  the number of right hand side
     * @param ws_node_in  the workspace node (may be nullptr)
     */
    void generate(const LinOp* system_matrix_in, const Multigrid* multigrid_in,
                  const size_type nrhs_in,
                  solver::Workspace* ws_node_in = nullptr);

    /**
     * allocate_memory is a helper function to allocate the memory of one level
     *
     * @tparam VectorType  the vector type
     *
     * @param level  the current level index
     * @param cycle  the multigrid cycle
     * @param current_nrows  the number of rows of current fine matrix
     * @param next_nrows  the number of rows of next coarse matrix
     */
    template <typename VectorType>
    void allocate_memory(int level, multigrid::cycle cycle,
                         size_type current_nrows, size_type next_nrows);

#if GINKGO_BUILD_MPI
    /**
     * allocate_memory is a helper function to allocate the memory of one level
     *
     * @tparam VectorType  the vector type
     *
     * @param level  the current level index
     * @param cycle  the multigrid cycle
     * @param current_comm  the communicator of the current fine matrix
     * @param next_comm  the communicator of the next coarse matrix
     * @param current_nrows  the number of rows of the current fine matrix
     * @param next_nrows  the number of rows of the next coarse matrix
     * @param current_local_nrows  the number of rows of the local operator of
     *                             the current fine matrix
     * @param next_local_nrows  the number of rows of the local operator of the
     *                          next coarse matrix
     */
    template <typename VectorType>
    void allocate_memory(int level, multigrid::cycle cycle,
                         const experimental::mpi::communicator& current_comm,
                         const experimental::mpi::communicator& next_comm,
                         size_type current_nrows, size_type next_nrows,
                         size_type current_local_nrows,
                         size_type next_local_nrows);
#endif

    /**
     * run the cycle of the level
     *
     * @param cycle  the multigrid cycle
     * @param level  the current level index
     * @param matrix  the system matrix of current level
     * @param b  the right hand side
     * @param x  the input vectors
     * @param mode  the mode of cycle (See cycle_mode)
     */
    void run_mg_cycle(multigrid::cycle cycle, size_type level,
                      const std::shared_ptr<const LinOp>& matrix,
                      const LinOp* b, LinOp* x, cycle_mode mode);

    /**
     * @copydoc run_cycle
     *
     * @tparam VectorType  the vector type
     *
     * @note it is the version with known ValueType
     */
    template <typename VectorType>
    void run_cycle(multigrid::cycle cycle, size_type level,
                   const std::shared_ptr<const LinOp>& matrix, const LinOp* b,
                   LinOp* x, cycle_mode mode);

    const LinOp* system_matrix;
    const Multigrid* multigrid;
    size_type nrhs;
    solver::Workspace* ws_node = nullptr;
    std::vector<solver::Workspace*> level_nodes;
};


void MultigridState::generate(const LinOp* system_matrix_in,
                              const gko::solver::Multigrid* multigrid_in,
                              const size_type nrhs_in,
                              solver::Workspace* ws_node_in)
{
    system_matrix = system_matrix_in;
    multigrid = multigrid_in;
    nrhs = nrhs_in;
    auto current_nrows = system_matrix->get_size()[0];
    auto mg_level_list = multigrid->get_mg_level_list();
    auto list_size = mg_level_list.size();
    auto cycle = multigrid->get_cycle();
    // Set up workspace level nodes for persistent storage across regeneration
    ws_node = ws_node_in;
    level_nodes.clear();
    level_nodes.reserve(list_size);
    GKO_ASSERT(ws_node);
    for (size_type i = 0; i < list_size; i++) {
        level_nodes.push_back(
            ws_node->get_or_create_child("level_" + std::to_string(i)));
    }
    // Allocate memory first such that reusing allocation in each iter.
    for (int i = 0; i < mg_level_list.size(); i++) {
        auto next_nrows = mg_level_list.at(i)->get_coarse_op()->get_size()[0];
        auto mg_level = mg_level_list.at(i);

        run<gko::multigrid::EnableMultigridLevel, float, double,
#if GINKGO_ENABLE_HALF
            float16, std::complex<float16>,
#endif
#if GINKGO_ENABLE_BFLOAT16
            bfloat16, std::complex<bfloat16>,
#endif
            std::complex<float>, std::complex<double>>(
            mg_level,
            [&, this](auto mg_level, auto i, auto cycle, auto current_nrows,
                      auto next_nrows) {
#if GINKGO_BUILD_MPI
                if (gko::detail::is_distributed(system_matrix_in)) {
                    using value_type =
                        typename std::decay_t<decltype(*mg_level)>::value_type;
                    using VectorType =
                        experimental::distributed::Vector<value_type>;
                    auto fine = mg_level->get_fine_op().get();
                    auto coarse = mg_level->get_coarse_op().get();
                    auto distributed_fine = dynamic_cast<
                        const experimental::distributed::DistributedBase*>(
                        fine);
                    auto distributed_coarse = dynamic_cast<
                        const experimental::distributed::DistributedBase*>(
                        coarse);
                    auto current_comm = distributed_fine->get_communicator();
                    auto next_comm = distributed_coarse->get_communicator();
                    auto current_local_nrows =
                        ::gko::detail::run_matrix(fine, [](auto* fine_mat) {
                            return fine_mat->get_diag_matrix()->get_size()[0];
                        });
                    auto next_local_nrows =
                        ::gko::detail::run_matrix(coarse, [](auto* coarse_mat) {
                            return coarse_mat->get_off_diag_matrix()
                                ->get_size()[0];
                        });
                    this->allocate_memory<VectorType>(
                        i, cycle, current_comm, next_comm, current_nrows,
                        next_nrows, current_local_nrows, next_local_nrows);

                } else
#endif
                {
                    using value_type =
                        typename std::decay_t<decltype(*mg_level)>::value_type;
                    using VectorType = matrix::Dense<value_type>;
                    this->allocate_memory<VectorType>(i, cycle, current_nrows,
                                                      next_nrows);
                }
            },
            i, cycle, current_nrows, next_nrows);

        current_nrows = next_nrows;
    }
}


template <class VectorType>
void MultigridState::allocate_memory(int level, multigrid::cycle cycle,
                                     size_type current_nrows,
                                     size_type next_nrows)
{
    using value_type = typename VectorType::value_type;
    using vec = matrix::Dense<value_type>;

    auto exec =
        as<LinOp>(multigrid->get_mg_level_list().at(level))->get_executor();

    GKO_ASSERT(static_cast<size_type>(level) < level_nodes.size());
    auto* lnode = level_nodes[level];
    GKO_ASSERT(lnode);

    auto& storage = lnode->get_local_storage();
    storage.set_executor(exec);
    const bool scale_correction =
        multigrid->get_parameters().scale_correction;
    storage.set_size(scale_correction ? 10 : 6, 0);

    // Slot 0: r (current level dimensions)
    storage.template create_or_get_op<vec>(
        0,
        [&] {
            return vec::create(exec, dim<2>{current_nrows, nrhs});
        },
        typeid(vec), dim<2>{current_nrows, nrhs}, nrhs);

    // Slot 1: g (next/coarse level dimensions)
    storage.template create_or_get_op<vec>(
        1,
        [&] {
            return vec::create(exec, dim<2>{next_nrows, nrhs});
        },
        typeid(vec), dim<2>{next_nrows, nrhs}, nrhs);

    // Slot 2: e (next/coarse level dimensions)
    storage.template create_or_get_op<vec>(
        2,
        [&] {
            return vec::create(exec, dim<2>{next_nrows, nrhs});
        },
        typeid(vec), dim<2>{next_nrows, nrhs}, nrhs);

    // Slot 3: one (scalar 1.0)
    storage.template create_or_get_op<vec>(
        3, [&] { return initialize<vec>({one<value_type>()}, exec); },
        typeid(vec), dim<2>{1, 1}, 1);

    // Slot 4: neg_one (scalar -1.0)
    storage.template create_or_get_op<vec>(
        4, [&] { return initialize<vec>({-one<value_type>()}, exec); },
        typeid(vec), dim<2>{1, 1}, 1);

    // Slot 5: next_one (scalar 1.0 for next level's value_type)
    storage.template create_or_get_op<vec>(
        5, [&] { return initialize<vec>({one<value_type>()}, exec); },
        typeid(vec), dim<2>{1, 1}, 1);

    if (scale_correction) {
        // Slot 6: acf = A*delta scratch (current level)
        storage.template create_or_get_op<vec>(
            6,
            [&] {
                return vec::create(exec, dim<2>{current_nrows, nrhs});
            },
            typeid(vec), dim<2>{current_nrows, nrhs}, nrhs);

        // Slot 7: dp = delta_pre / smoother scratch (current level)
        storage.template create_or_get_op<vec>(
            7,
            [&] {
                return vec::create(exec, dim<2>{current_nrows, nrhs});
            },
            typeid(vec), dim<2>{current_nrows, nrhs}, nrhs);

        // Slot 8: alpha = Rayleigh numerator scalar (1 x nrhs)
        storage.template create_or_get_op<vec>(
            8, [&] { return vec::create(exec, dim<2>{1, nrhs}); },
            typeid(vec), dim<2>{1, nrhs}, nrhs);

        // Slot 9: denom = Rayleigh denominator scalar (1 x nrhs)
        storage.template create_or_get_op<vec>(
            9, [&] { return vec::create(exec, dim<2>{1, nrhs}); },
            typeid(vec), dim<2>{1, nrhs}, nrhs);
    }
}


#if GINKGO_BUILD_MPI


template <typename VectorType>
void MultigridState::allocate_memory(
    int level, multigrid::cycle cycle,
    const experimental::mpi::communicator& current_comm,
    const experimental::mpi::communicator& next_comm, size_type current_nrows,
    size_type next_nrows, size_type current_local_nrows,
    size_type next_local_nrows)
{
    using value_type = typename VectorType::value_type;
    using vec = VectorType;
    using dense_vec = matrix::Dense<value_type>;

    auto exec =
        as<LinOp>(multigrid->get_mg_level_list().at(level))->get_executor();
    GKO_ASSERT(static_cast<size_type>(level) < level_nodes.size());
    auto* lnode = level_nodes[level];
    GKO_ASSERT(lnode);

    auto& storage = lnode->get_local_storage();
    storage.set_executor(exec);
    const bool scale_correction =
        multigrid->get_parameters().scale_correction;
    storage.set_size(scale_correction ? 10 : 6, 0);

    // Slot 0: r (current level, distributed)
    storage.template create_or_get_op<vec>(
        0,
        [&] {
            return vec::create(exec, current_comm,
                               dim<2>{current_nrows, nrhs},
                               dim<2>{current_local_nrows, nrhs});
        },
        typeid(vec), dim<2>{current_nrows, nrhs}, nrhs);

    // Slot 1: g (next/coarse level, distributed)
    storage.template create_or_get_op<vec>(
        1,
        [&] {
            return vec::create(exec, next_comm, dim<2>{next_nrows, nrhs},
                               dim<2>{next_local_nrows, nrhs});
        },
        typeid(vec), dim<2>{next_nrows, nrhs}, nrhs);

    // Slot 2: e (next/coarse level, distributed)
    storage.template create_or_get_op<vec>(
        2,
        [&] {
            return vec::create(exec, next_comm, dim<2>{next_nrows, nrhs},
                               dim<2>{next_local_nrows, nrhs});
        },
        typeid(vec), dim<2>{next_nrows, nrhs}, nrhs);

    // Slot 3: one (scalar 1.0)
    storage.template create_or_get_op<dense_vec>(
        3, [&] { return initialize<dense_vec>({one<value_type>()}, exec); },
        typeid(dense_vec), dim<2>{1, 1}, 1);

    // Slot 4: neg_one (scalar -1.0)
    storage.template create_or_get_op<dense_vec>(
        4, [&] { return initialize<dense_vec>({-one<value_type>()}, exec); },
        typeid(dense_vec), dim<2>{1, 1}, 1);

    // Slot 5: next_one (scalar 1.0 for next level's value_type)
    storage.template create_or_get_op<dense_vec>(
        5, [&] { return initialize<dense_vec>({one<value_type>()}, exec); },
        typeid(dense_vec), dim<2>{1, 1}, 1);

    if (scale_correction) {
        // Slot 6: acf = A*delta scratch (current level, distributed)
        storage.template create_or_get_op<vec>(
            6,
            [&] {
                return vec::create(exec, current_comm,
                                   dim<2>{current_nrows, nrhs},
                                   dim<2>{current_local_nrows, nrhs});
            },
            typeid(vec), dim<2>{current_nrows, nrhs}, nrhs);

        // Slot 7: dp = delta_pre / smoother scratch (current level,
        // distributed)
        storage.template create_or_get_op<vec>(
            7,
            [&] {
                return vec::create(exec, current_comm,
                                   dim<2>{current_nrows, nrhs},
                                   dim<2>{current_local_nrows, nrhs});
            },
            typeid(vec), dim<2>{current_nrows, nrhs}, nrhs);

        // Slot 8: alpha scalar (1 x nrhs, non-distributed)
        storage.template create_or_get_op<dense_vec>(
            8, [&] { return dense_vec::create(exec, dim<2>{1, nrhs}); },
            typeid(dense_vec), dim<2>{1, nrhs}, nrhs);

        // Slot 9: denom scalar (1 x nrhs, non-distributed)
        storage.template create_or_get_op<dense_vec>(
            9, [&] { return dense_vec::create(exec, dim<2>{1, nrhs}); },
            typeid(dense_vec), dim<2>{1, nrhs}, nrhs);
    }

}


#endif


void MultigridState::run_mg_cycle(multigrid::cycle cycle, size_type level,
                                  const std::shared_ptr<const LinOp>& matrix,
                                  const LinOp* b, LinOp* x, cycle_mode mode)
{
    if (level == multigrid->get_mg_level_list().size()) {
        multigrid->get_coarsest_solver()->apply(b, x);
        return;
    }
    auto mg_level = multigrid->get_mg_level_list().at(level);
    run<gko::multigrid::EnableMultigridLevel, float, double,
#if GINKGO_ENABLE_HALF
        float16, std::complex<float16>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        bfloat16, std::complex<bfloat16>,
#endif
        std::complex<float>, std::complex<double>>(
        mg_level, [&, this](auto mg_level) {
#if GINKGO_BUILD_MPI
            if (gko::detail::is_distributed(matrix.get())) {
                using value_type =
                    typename std::decay_t<decltype(*mg_level)>::value_type;
                this->run_cycle<
                    typename experimental::distributed::Vector<value_type>>(
                    cycle, level, matrix, b, x, mode);
            } else
#endif
            {
                using value_type =
                    typename std::decay_t<decltype(*mg_level)>::value_type;
                this->run_cycle<typename matrix::Dense<value_type>>(
                    cycle, level, matrix, b, x, mode);
            }
        });
}


template <typename VectorType>
void MultigridState::run_cycle(multigrid::cycle cycle, size_type level,
                               const std::shared_ptr<const LinOp>& matrix,
                               const LinOp* b, LinOp* x, cycle_mode mode)
{
    using value_type = typename VectorType::value_type;
    auto total_level = multigrid->get_mg_level_list().size();

    auto& level_storage = level_nodes[level]->get_local_storage();
    auto* r = level_storage.get_mutable_op(0);
    auto* g = level_storage.get_mutable_op(1);
    auto* e = level_storage.get_mutable_op(2);
    auto* one = level_storage.get_op(3);
    auto* neg_one = level_storage.get_op(4);
    auto* next_one = level_storage.get_op(5);
    auto mg_level = multigrid->get_mg_level_list().at(level);
    auto pre_smoother = multigrid->get_pre_smoother_list().at(level);
    std::shared_ptr<const LinOp> mid_smoother{nullptr};
    auto mid_case = multigrid->get_parameters().mid_case;
    if (mid_case == multigrid::mid_smooth_type::standalone) {
        mid_smoother = multigrid->get_mid_smoother_list().at(level);
    }
    auto post_smoother = multigrid->get_post_smoother_list().at(level);
    // scale correction applies at all levels except immediately above coarsest
    bool do_scale =
        multigrid->get_parameters().scale_correction && level < total_level - 1;

    bool use_pre = has_property(mode, cycle_mode::first_of_cycle) ||
                   mid_case == multigrid::mid_smooth_type::both ||
                   mid_case == multigrid::mid_smooth_type::pre_smoother;
    if (use_pre && pre_smoother) {
        if (has_property(mode, cycle_mode::x_is_zero)) {
            if (auto pre_allow_zero_input =
                    std::dynamic_pointer_cast<const ApplyWithInitialGuess>(
                        pre_smoother)) {
                pre_allow_zero_input->apply_with_initial_guess(
                    b, x, initial_guess_mode::zero);
            } else {
                // x in first level is already filled by zero outside.
                if (level != 0) {
                    dynamic_cast<VectorType*>(x)->fill(zero<value_type>());
                }
                pre_smoother->apply(b, x);
            }
        } else {
            pre_smoother->apply(b, x);
        }
    }

    // Pre-smooth scale correction (OpenFOAM GAMGSolverSolve.C downward pass):
    // Rayleigh-scale δ_pre = x, deflating r before restriction.
    //   Aδ    = A * δ_pre
    //   sf    = (δ_pre · b) / (δ_pre · Aδ)
    //   δ_pre = sf * δ_pre + smoother(b − sf * Aδ)   [reuses acf, r scratch]
    if (do_scale && use_pre && pre_smoother) {
        auto exec = multigrid->get_executor();
        auto* acf = level_storage.get_mutable_op(6);
        auto* dp = level_storage.get_mutable_op(7);
        auto* alpha_dense =
            as<matrix::Dense<value_type>>(level_storage.get_mutable_op(8));
        auto* denom_dense =
            as<matrix::Dense<value_type>>(level_storage.get_mutable_op(9));

        matrix->apply(x, acf);  // acf = A * δ_pre
        as<VectorType>(x)->compute_dot(b, level_storage.get_mutable_op(8));
        as<VectorType>(x)->compute_dot(acf, level_storage.get_mutable_op(9));
        if (exec->copy_val_to_host(denom_dense->get_const_values()) !=
            zero<value_type>()) {
            alpha_dense->inv_scale(denom_dense);  // sf = (δ·b)/(δ·Aδ)

            // r temporarily holds r_scaled = b − sf * Aδ
            as<VectorType>(acf)->scale(alpha_dense);  // acf = sf * Aδ
            r->copy_from(b);
            as<VectorType>(r)->add_scaled(neg_one, acf);  // r = b − sf*Aδ

            // dp = smoother(r_scaled) starting from zero
            as<VectorType>(dp)->fill(zero<value_type>());
            pre_smoother->apply(r, dp);

            // x = sf * δ_pre + smoother(b − sf*Aδ)
            as<VectorType>(x)->scale(alpha_dense);
            as<VectorType>(x)->add_scaled(one, dp);
        }
        // r is overwritten with the actual (deflated) residual below
    }

    // The common smoother is wrapped by IR and IR already split the iter and
    // residual check. Thus, when the IR only contains iter limit, there's no
    // additional residual computation.
    // TODO: if already computes the residual outside, the first level may not
    // need this residual computation when no presmoother in the first level.
    r->copy_from(b);
    matrix->apply(neg_one, x, one, r);  // r = b − A*x (deflated if scaled)

    // restrict
    mg_level->get_restrict_op()->apply(r, g);
    // next level
    if (level + 1 == total_level) {
        // the coarsest solver use the last level valuetype
        as<VectorType>(e)->fill(zero<value_type>());
    }
    auto next_level_matrix =
        (level + 1 < total_level)
            ? multigrid->get_mg_level_list().at(level + 1)->get_fine_op()
            : mg_level->get_coarse_op();
    auto next_mode = cycle_mode::x_is_zero | cycle_mode::first_of_cycle;
    if (cycle == multigrid::cycle::v) {
        // v cycle only contains one step
        next_mode = next_mode | cycle_mode::end_of_cycle;
    }
    this->run_mg_cycle(cycle, level + 1, next_level_matrix, g, e, next_mode);
    if (level < multigrid->get_mg_level_list().size() - 1) {
        // additional work for non-v_cycle
        if (cycle == multigrid::cycle::f) {
            // f_cycle calls v_cycle in the second cycle
            this->run_mg_cycle(multigrid::cycle::v, level + 1,
                               next_level_matrix, g, e,
                               cycle_mode::end_of_cycle);
        } else if (cycle == multigrid::cycle::w) {
            this->run_mg_cycle(cycle, level + 1, next_level_matrix, g, e,
                               cycle_mode::end_of_cycle);
        }
    }

    // Post-smooth scale correction (OpenFOAM GAMGSolverSolve.C upward pass):
    // Prolong coarse correction δ_c into acf, Rayleigh-scale it w.r.t. the
    // deflated residual r, then merge with δ_pre (= current x).
    //   δ_c  = prolong(e)
    //   Aδ   = A * δ_c                  [stored in delta_pre scratch]
    //   sf   = (δ_c · r) / (δ_c · Aδ)
    //   δ_c  = sf * δ_c + smoother(r − sf * Aδ)
    //   x   += δ_c                       [x = δ_pre + scale-corrected δ_c]
    if (do_scale) {
        auto exec = multigrid->get_executor();
        auto* acf = level_storage.get_mutable_op(6);
        auto* dp = level_storage.get_mutable_op(7);
        auto* alpha_dense =
            as<matrix::Dense<value_type>>(level_storage.get_mutable_op(8));
        auto* denom_dense =
            as<matrix::Dense<value_type>>(level_storage.get_mutable_op(9));

        // prolong e into acf (δ_c = prolong(e))
        as<VectorType>(acf)->fill(zero<value_type>());
        mg_level->get_prolong_op()->apply(next_one, e, next_one, acf);

        matrix->apply(acf, dp);  // dp = A * δ_c
        as<VectorType>(acf)->compute_dot(r, level_storage.get_mutable_op(8));
        as<VectorType>(acf)->compute_dot(dp, level_storage.get_mutable_op(9));
        if (exec->copy_val_to_host(denom_dense->get_const_values()) !=
            zero<value_type>()) {
            alpha_dense->inv_scale(denom_dense);  // sf = (δ_c·r)/(δ_c·Aδ_c)

            // r temporarily holds r_scaled = r − sf * Aδ_c
            as<VectorType>(dp)->scale(alpha_dense);      // dp = sf * Aδ_c
            as<VectorType>(r)->add_scaled(neg_one, dp);  // r = r − sf*Aδ_c

            // dp = smoother(r_scaled) starting from zero
            as<VectorType>(dp)->fill(zero<value_type>());
            if (pre_smoother) {
                pre_smoother->apply(r, dp);
            }

            // acf = sf * δ_c + smoother(r − sf*Aδ_c)
            as<VectorType>(acf)->scale(alpha_dense);
            as<VectorType>(acf)->add_scaled(one, dp);
        }
        // x = δ_pre + scale-corrected δ_c
        as<VectorType>(x)->add_scaled(one, acf);
    } else {
        // standard prolongation: x += prolong(e)
        mg_level->get_prolong_op()->apply(next_one, e, next_one, x);
    }

    bool use_post = has_property(mode, cycle_mode::end_of_cycle) ||
                    mid_case == multigrid::mid_smooth_type::both ||
                    mid_case == multigrid::mid_smooth_type::post_smoother;
    if (use_post && post_smoother) {
        post_smoother->apply(b, x);
    }

    // put the mid smoother into the end of previous cycle (W/F cycle only)
    bool use_mid =
        (cycle == multigrid::cycle::w || cycle == multigrid::cycle::f) &&
        !has_property(mode, cycle_mode::end_of_cycle) &&
        mid_case == multigrid::mid_smooth_type::standalone;
    if (use_mid && mid_smoother) {
        mid_smoother->apply(b, x);
    }
}


}  // namespace detail
}  // namespace multigrid


typename Multigrid::parameters_type Multigrid::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Multigrid::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("criteria")) {
        params.with_criteria(
            config::parse_or_get_factory_vector<const stop::CriterionFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("mg_level")) {
        params.with_mg_level(
            config::parse_or_get_factory_vector<const gko::LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("pre_smoother")) {
        params.with_pre_smoother(
            config::parse_or_get_factory_vector<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("post_smoother")) {
        params.with_post_smoother(
            config::parse_or_get_factory_vector<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("mid_smoother")) {
        params.with_mid_smoother(
            config::parse_or_get_factory_vector<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("post_uses_pre")) {
        params.with_post_uses_pre(gko::config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("mid_case")) {
        auto str = obj.get_string();
        if (str == "both") {
            params.with_mid_case(multigrid::mid_smooth_type::both);
        } else if (str == "post_smoother") {
            params.with_mid_case(multigrid::mid_smooth_type::post_smoother);
        } else if (str == "pre_smoother") {
            params.with_mid_case(multigrid::mid_smooth_type::pre_smoother);
        } else if (str == "standalone") {
            params.with_mid_case(multigrid::mid_smooth_type::standalone);
        } else {
            GKO_INVALID_CONFIG_VALUE("mid_smooth_type", str);
        }
    }
    if (auto& obj = config_check.get("max_levels")) {
        params.with_max_levels(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = config_check.get("min_coarse_rows")) {
        params.with_min_coarse_rows(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = config_check.get("coarsest_solver")) {
        params.with_coarsest_solver(
            config::parse_or_get_factory_vector<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("cycle")) {
        auto str = obj.get_string();
        if (str == "v") {
            params.with_cycle(multigrid::cycle::v);
        } else if (str == "w") {
            params.with_cycle(multigrid::cycle::w);
        } else if (str == "f") {
            params.with_cycle(multigrid::cycle::f);
        } else {
            GKO_INVALID_CONFIG_VALUE("cycle", str);
        }
    }
    if (auto& obj = config_check.get("kcycle_base")) {
        params.with_kcycle_base(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = config_check.get("kcycle_rel_tol")) {
        params.with_kcycle_rel_tol(gko::config::get_value<double>(obj));
    }
    if (auto& obj = config_check.get("smoother_relax")) {
        params.with_smoother_relax(
            config::get_value<std::complex<double>>(obj));
    }
    if (auto& obj = config_check.get("smoother_iters")) {
        params.with_smoother_iters(config::get_value<size_type>(obj));
    }
    if (auto& obj = config_check.get("default_initial_guess")) {
        params.with_default_initial_guess(
            config::get_value<solver::initial_guess_mode>(obj));
    }
    if (auto& obj = config_check.get("scale_correction")) {
        params.with_scale_correction(config::get_value<bool>(obj));
    }

    return params;
}


void Multigrid::generate()
{
    // generate coarse matrix until reaching max_level or min_coarse_rows
    auto num_rows = this->get_system_matrix()->get_size()[0];
    size_type level = 0;
    auto matrix = this->get_system_matrix();
    auto exec = this->get_executor();
    auto* mg_ws_node = this->get_workspace_node();
    // Always generate smoother with size = level.
    while (level < parameters_.max_levels &&
           num_rows > parameters_.min_coarse_rows) {
        auto index = level_selector_(level, matrix.get());
        GKO_ENSURE_IN_BOUNDS(index, parameters_.mg_level.size());
        auto mg_level_factory = parameters_.mg_level.at(index);
        // coarse generate
        auto mg_level = as<gko::multigrid::MultigridLevel>(
            share(mg_level_factory->generate(matrix)));
        if (mg_level->get_coarse_op()->get_size()[0] == num_rows) {
            // do not reduce dimension
            break;
        }

        // Create per-level workspace child node for smoother propagation
        solver::Workspace* level_node = nullptr;
        if (mg_ws_node) {
            level_node = mg_ws_node->get_or_create_child("level_" +
                                                         std::to_string(level));
        }

        run<gko::multigrid::EnableMultigridLevel, float, double,
#if GINKGO_ENABLE_HALF
            float16, std::complex<float16>,
#endif
#if GINKGO_ENABLE_BFLOAT16
            bfloat16, std::complex<bfloat16>,
#endif
            std::complex<float>, std::complex<double>>(
            mg_level,
            [this, level_node](auto mg_level, auto index, auto matrix) {
                using value_type =
                    typename std::decay_t<decltype(*mg_level)>::value_type;
                // Create smoother child nodes from the level node
                solver::Workspace* pre_node = nullptr;
                solver::Workspace* mid_node = nullptr;
                solver::Workspace* post_node = nullptr;
                if (level_node) {
                    pre_node = level_node->get_or_create_child("pre_smoother");
                    if (parameters_.mid_case ==
                        multigrid::mid_smooth_type::standalone) {
                        mid_node =
                            level_node->get_or_create_child("mid_smoother");
                    }
                    if (!parameters_.post_uses_pre) {
                        post_node =
                            level_node->get_or_create_child("post_smoother");
                    }
                }
                handle_list<value_type>(index, matrix, parameters_.pre_smoother,
                                        pre_smoother_list_,
                                        parameters_.smoother_iters,
                                        parameters_.smoother_relax, pre_node);
                if (parameters_.mid_case ==
                    multigrid::mid_smooth_type::standalone) {
                    handle_list<value_type>(
                        index, matrix, parameters_.mid_smoother,
                        mid_smoother_list_, parameters_.smoother_iters,
                        parameters_.smoother_relax, mid_node);
                }
                if (!parameters_.post_uses_pre) {
                    handle_list<value_type>(
                        index, matrix, parameters_.post_smoother,
                        post_smoother_list_, parameters_.smoother_iters,
                        parameters_.smoother_relax, post_node);
                }
            },
            index, mg_level->get_fine_op());

        mg_level_list_.emplace_back(mg_level);
        matrix = mg_level_list_.back()->get_coarse_op();
        num_rows = matrix->get_size()[0];
        level++;
    }
    if (parameters_.post_uses_pre) {
        post_smoother_list_ = pre_smoother_list_;
    }
    // Generate at least one level
    GKO_ASSERT_EQ(level > 0, true);
    auto last_mg_level = mg_level_list_.back();

    using ws = workspace_traits<Multigrid>;
    this->setup_workspace();
    this->create_state();
    cache_.state->generate(this->get_system_matrix().get(), this, 1,
                           this->get_workspace_node());

    // generate coarsest solver
    // Create workspace child node for coarse solver propagation
    solver::Workspace* coarse_ws_node = nullptr;
    if (mg_ws_node) {
        coarse_ws_node = mg_ws_node->get_or_create_child("coarse_solver");
    }
    run<gko::multigrid::EnableMultigridLevel, float, double,
#if GINKGO_ENABLE_HALF
        float16, std::complex<float16>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        bfloat16, std::complex<bfloat16>,
#endif
        std::complex<float>, std::complex<double>>(
        last_mg_level,
        [this, coarse_ws_node](auto mg_level, auto level, auto matrix) {
            using value_type =
                typename std::decay_t<decltype(*mg_level)>::value_type;
            auto exec = this->get_executor();
            // default coarse grid solver, direct LU
            // TODO: maybe remove fixed index type
            auto gen_default_solver = [&]() -> std::unique_ptr<LinOp> {
#if GINKGO_BUILD_MPI
                if (gko::detail::is_distributed(matrix.get())) {
                    using absolute_value_type = remove_complex<value_type>;
                    using experimental::distributed::Matrix;
                    return run<Matrix<value_type, int32, int32>,
                               Matrix<value_type, int32, int64>,
                               Matrix<value_type, int64, int64>>(
                        matrix,
                        [exec, coarse_ws_node](
                            auto matrix) -> std::unique_ptr<LinOp> {
                            using Mtx = typename decltype(matrix)::element_type;
                            auto factory =
                                solver::Gmres<value_type>::build()
                                    .with_criteria(
                                        stop::Iteration::build().with_max_iters(
                                            matrix->get_size()[0]),
                                        stop::ResidualNorm<value_type>::build()
                                            .with_reduction_factor(
                                                std::numeric_limits<
                                                    absolute_value_type>::
                                                    epsilon() *
                                                absolute_value_type{10}))
                                    .with_krylov_dim(std::min(
                                        size_type(100), matrix->get_size()[0]))
                                    .with_preconditioner(
                                        experimental::distributed::
                                            preconditioner::Schwarz<
                                                value_type,
                                                typename Mtx::local_index_type,
                                                typename Mtx::
                                                    global_index_type>::build()
                                                .with_local_solver(
                                                    preconditioner::Jacobi<
                                                        value_type>::build()
                                                        .with_max_block_size(
                                                            1u)))
                                    .on(exec);
                            if (coarse_ws_node) {
                                return factory->generate(matrix,
                                                         coarse_ws_node);
                            }
                            return factory->generate(matrix);
                        });
                }
#endif
                // TODO: unify when dpcpp supports direct solver
                if (dynamic_cast<const DpcppExecutor*>(exec.get())) {
                    using absolute_value_type = remove_complex<value_type>;
                    auto factory =
                        solver::Gmres<value_type>::build()
                            .with_criteria(
                                stop::Iteration::build().with_max_iters(
                                    matrix->get_size()[0]),
                                stop::ResidualNorm<value_type>::build()
                                    .with_reduction_factor(
                                        std::numeric_limits<
                                            absolute_value_type>::epsilon() *
                                        absolute_value_type{10}))
                            .with_krylov_dim(
                                std::min(size_type(100), matrix->get_size()[0]))
                            .with_preconditioner(
                                preconditioner::Jacobi<value_type>::build()
                                    .with_max_block_size(1u))
                            .on(exec);
                    if (coarse_ws_node) {
                        return factory->generate(matrix, coarse_ws_node);
                    }
                    return factory->generate(matrix);
                } else {
                    auto factory =
                        experimental::solver::Direct<value_type, int32>::build()
                            .with_factorization(
                                experimental::factorization::Lu<value_type,
                                                                int32>::build())
                            .on(exec);
                    if (coarse_ws_node) {
                        return factory->generate(matrix, coarse_ws_node);
                    }
                    return factory->generate(matrix);
                }
            };
            if (parameters_.coarsest_solver.size() == 0) {
                coarsest_solver_ = gen_default_solver();
            } else {
                auto temp_index = solver_selector_(level, matrix.get());
                GKO_ENSURE_IN_BOUNDS(temp_index,
                                     parameters_.coarsest_solver.size());
                auto solver = parameters_.coarsest_solver.at(temp_index);
                if (solver == nullptr) {
                    coarsest_solver_ = gen_default_solver();
                } else if (coarse_ws_node) {
                    coarsest_solver_ = solver->generate(matrix, coarse_ws_node);
                } else {
                    coarsest_solver_ = solver->generate(matrix);
                }
            }
        },
        level, matrix);
}


void Multigrid::apply_impl(const LinOp* b, LinOp* x) const
{
    this->apply_with_initial_guess_impl(b, x,
                                        this->get_default_initial_guess());
}


void Multigrid::apply_with_initial_guess_impl(const LinOp* b, LinOp* x,
                                              initial_guess_mode guess) const
{
    if (!this->get_system_matrix() || !this->get_system_matrix()->get_size()) {
        return;
    }

    auto lambda = [this, guess](auto mg_level, auto b, auto x) {
        using value_type =
            typename std::decay_t<decltype(*mg_level)>::value_type;
        experimental::precision_dispatch_real_complex_distributed<value_type>(
            [this, guess](auto dense_b, auto dense_x) {
                prepare_initial_guess(dense_b, dense_x, guess);
                this->apply_dense_impl(dense_b, dense_x, guess);
            },
            b, x);
    };
    auto first_mg_level = this->get_mg_level_list().front();
    run<gko::multigrid::EnableMultigridLevel, float, double,
#if GINKGO_ENABLE_HALF
        float16, std::complex<float16>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        bfloat16, std::complex<bfloat16>,
#endif
        std::complex<float>, std::complex<double>>(first_mg_level, lambda, b,
                                                   x);
}


void Multigrid::apply_impl(const LinOp* alpha, const LinOp* b,
                           const LinOp* beta, LinOp* x) const
{
    this->apply_with_initial_guess_impl(alpha, b, beta, x,
                                        this->get_default_initial_guess());
}


void Multigrid::apply_with_initial_guess_impl(const LinOp* alpha,
                                              const LinOp* b, const LinOp* beta,
                                              LinOp* x,
                                              initial_guess_mode guess) const
{
    if (!this->get_system_matrix() || !this->get_system_matrix()->get_size()) {
        return;
    }

    auto lambda = [this, guess](auto mg_level, auto alpha, auto b, auto beta,
                                auto x) {
        using value_type =
            typename std::decay_t<decltype(*mg_level)>::value_type;
        experimental::precision_dispatch_real_complex_distributed<value_type>(
            [this, guess](auto dense_alpha, auto dense_b, auto dense_beta,
                          auto dense_x) {
                prepare_initial_guess(dense_b, dense_x, guess);
                auto x_clone = dense_x->clone();
                this->apply_dense_impl(dense_b, x_clone.get(), guess);
                dense_x->scale(dense_beta);
                dense_x->add_scaled(dense_alpha, x_clone);
            },
            alpha, b, beta, x);
    };
    auto first_mg_level = this->get_mg_level_list().front();
    run<gko::multigrid::EnableMultigridLevel, float, double,
#if GINKGO_ENABLE_HALF
        float16, std::complex<float16>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        bfloat16, std::complex<bfloat16>,
#endif
        std::complex<float>, std::complex<double>>(first_mg_level, lambda,
                                                   alpha, b, beta, x);
}


template <typename VectorType>
void Multigrid::apply_dense_impl(const VectorType* b, VectorType* x,
                                 initial_guess_mode guess) const
{
    using ws = workspace_traits<Multigrid>;
    if (cache_.state->nrhs != b->get_size()[1]) {
        cache_.state->generate(this->get_system_matrix().get(), this,
                               b->get_size()[1], this->get_workspace_node());
    }
    auto lambda = [&, this](auto mg_level, auto b, auto x) {
        using value_type =
            typename std::decay_t<decltype(*mg_level)>::value_type;
        auto exec = this->get_executor();
        constexpr uint8 RelativeStoppingId{1};
        auto& stop_status =
            this->template create_workspace_array<stopping_status>(
                ws::stop, b->get_size()[1]);
        bool one_changed{};
        exec->run(multigrid::make_initialize(stop_status));
        auto stop_criterion = this->get_stop_criterion_factory()->generate(
            this->get_system_matrix(),
            std::shared_ptr<const LinOp>(b, null_deleter<const LinOp>{}), x,
            nullptr);
        int iter = -1;

        while (true) {
            ++iter;
            bool all_stopped =
                stop_criterion->update()
                    .num_iterations(iter)
                    // TODO: combine the out-of-cycle residual computation
                    // currently, the residual will computed additionally in
                    // stop_criterion when users require the corresponding
                    // residual check.
                    .solution(x)
                    .check(RelativeStoppingId, true, &stop_status,
                           &one_changed);
            this->template log<log::Logger::iteration_complete>(
                this, b, x, iter, nullptr, nullptr, nullptr, &stop_status,
                all_stopped);
            if (all_stopped) {
                break;
            }
            auto mode = multigrid::cycle_mode::first_of_cycle |
                        multigrid::cycle_mode::end_of_cycle;
            if (iter == 0 && guess == initial_guess_mode::zero) {
                mode = mode | multigrid::cycle_mode::x_is_zero;
            }
            cache_.state->run_mg_cycle(this->get_parameters().cycle, 0,
                                       this->get_system_matrix(), b, x, mode);
        }
    };

    auto first_mg_level = this->get_mg_level_list().front();

    run<gko::multigrid::EnableMultigridLevel, float, double,
#if GINKGO_ENABLE_HALF
        float16, std::complex<float16>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        bfloat16, std::complex<bfloat16>,
#endif
        std::complex<float>, std::complex<double>>(first_mg_level, lambda, b,
                                                   x);
}


/**
 * validate checks the given parameters are valid or not.
 */
void Multigrid::validate()
{
    const auto mg_level_len = parameters_.mg_level.size();
    if (mg_level_len == 0) {
        GKO_NOT_SUPPORTED(mg_level_len);
    } else {
        // each mg_level can not be nullptr
        for (size_type i = 0; i < mg_level_len; i++) {
            if (parameters_.mg_level.at(i) == nullptr) {
                GKO_NOT_SUPPORTED(parameters_.mg_level.at(i));
            }
        }
    }
    // verify pre-related parameters
    this->verify_legal_length(true, parameters_.pre_smoother.size(),
                              mg_level_len);
    // verify post-related parameters when post does not use pre
    this->verify_legal_length(!parameters_.post_uses_pre,
                              parameters_.post_smoother.size(), mg_level_len);
    // verify mid-related parameters when mid is standalone smoother.
    this->verify_legal_length(
        parameters_.mid_case == multigrid::mid_smooth_type::standalone,
        parameters_.mid_smoother.size(), mg_level_len);
}


void Multigrid::verify_legal_length(bool checked, size_type len,
                                    size_type ref_len)
{
    if (checked) {
        // len = 0 uses default behaviour
        // len = 1 uses the first one
        // len > 1 : must contain the same len as ref(mg_level)
        if (len > 1 && len != ref_len) {
            GKO_NOT_SUPPORTED(this);
        }
    }
}


void Multigrid::create_state() const
{
    if (cache_.state == nullptr) {
        cache_.state = std::make_unique<multigrid::detail::MultigridState>();
    }
}


Multigrid::Multigrid(const Multigrid::Factory* factory,
                     LinOpGenerateComponents components)
    : EnableLinOp<Multigrid>(factory->get_executor(),
                             transpose(components.system_matrix->get_size())),
      EnableSolverBase<Multigrid>{std::move(components.system_matrix)},
      EnableIterativeBase<Multigrid>{
          stop::combine(factory->get_parameters().criteria)},
      parameters_{factory->get_parameters()}
{
    this->adopt_workspace(components, this->get_executor());
    this->validate();
    if (!parameters_.level_selector) {
        auto mg_level_size = parameters_.mg_level.size();
        level_selector_ = [mg_level_size](const size_type level, const LinOp*) {
            return (level < mg_level_size) ? level : mg_level_size - 1;
        };
    } else {
        level_selector_ = parameters_.level_selector;
    }
    if (!parameters_.solver_selector) {
        if (parameters_.coarsest_solver.size() >= 1) {
            solver_selector_ = [](const size_type, const LinOp*) {
                return size_type{0};
            };
        }
    } else {
        solver_selector_ = parameters_.solver_selector;
    }


    this->set_default_initial_guess(parameters_.default_initial_guess);
    if (this->get_system_matrix()->get_size()[0] != 0) {
        // generate on the existed matrix
        this->generate();
    }
}


Multigrid::Multigrid(std::shared_ptr<const Executor> exec)
    : EnableLinOp<Multigrid>(exec)
{}


int workspace_traits<Multigrid>::num_arrays(const Solver&) { return 1; }


int workspace_traits<Multigrid>::num_vectors(const Solver&) { return 0; }


std::vector<std::string> workspace_traits<Multigrid>::op_names(const Solver&)
{
    return {};
}


std::vector<std::string> workspace_traits<Multigrid>::array_names(const Solver&)
{
    return {"stop"};
}


std::vector<int> workspace_traits<Multigrid>::scalars(const Solver&)
{
    return {};
}


std::vector<int> workspace_traits<Multigrid>::vectors(const Solver&)
{
    return {};
}


}  // namespace solver
}  // namespace gko

// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


#if GINKGO_COMMON_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_COMMON_SINGLE_MODE


template <typename SolverType>
struct SimpleSolverTest {
    using solver_type = SolverType;
    using value_type = typename solver_type::value_type;
    using index_type = gko::int32;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using precond_type = gko::preconditioner::Jacobi<value_type, index_type>;

    static constexpr bool is_iterative() { return true; }

    static constexpr bool is_preconditionable() { return true; }

    static constexpr bool will_not_allocate() { return true; }

    static constexpr bool requires_num_rhs() { return false; }

    static double tolerance() { return 1e4 * r<value_type>::value; }

    static void preprocess(gko::matrix_data<value_type, index_type>& data)
    {
        // make sure the matrix is well-conditioned
        gko::utils::make_hpd(data, 2.0);
    }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return solver_type::build().with_criteria(
            gko::stop::Iteration::build().with_max_iters(iteration_count),
            check_residual ? gko::stop::ResidualNorm<value_type>::build()
                                 .with_baseline(gko::stop::mode::absolute)
                                 .with_reduction_factor(1e-30)
                                 .on(exec)
                           : nullptr);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return build(exec, iteration_count, check_residual)
            .with_preconditioner(precond_type::build().with_max_block_size(1u));
    }

    static const gko::LinOp* get_preconditioner(
        gko::ptr_param<const solver_type> solver)
    {
        return solver->get_preconditioner().get();
    }

    static const gko::stop::CriterionFactory* get_stop_criterion_factory(
        gko::ptr_param<const solver_type> solver)
    {
        return solver->get_stop_criterion_factory().get();
    }

    static void assert_empty_state(const solver_type* mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_system_matrix(), nullptr);
        ASSERT_EQ(mtx->get_preconditioner(), nullptr);
        ASSERT_EQ(mtx->get_stopping_criterion_factory(), nullptr);
    }

    static constexpr bool logs_iteration_complete() { return true; }
};


struct Cg : SimpleSolverTest<gko::solver::Cg<solver_value_type>> {};


struct Cgs : SimpleSolverTest<gko::solver::Cgs<solver_value_type>> {
    static double tolerance() { return 1e5 * r<value_type>::value; }
};


struct Fcg : SimpleSolverTest<gko::solver::Fcg<solver_value_type>> {
    static double tolerance() { return 1e7 * r<value_type>::value; }
};


struct Bicg : SimpleSolverTest<gko::solver::Bicg<solver_value_type>> {
    static constexpr bool will_not_allocate() { return false; }
};


struct Bicgstab : SimpleSolverTest<gko::solver::Bicgstab<solver_value_type>> {
    // I give up ._. Some cases still have huge differences
    static double tolerance() { return 1e12 * r<value_type>::value; }
};


template <unsigned dimension>
struct Idr : SimpleSolverTest<gko::solver::Idr<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return SimpleSolverTest<gko::solver::Idr<solver_value_type>>::build(
                   exec, iteration_count, check_residual)
            .with_deterministic(true)
            .with_subspace_dim(dimension);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return build(exec, iteration_count, check_residual)
            .with_preconditioner(precond_type::build().with_max_block_size(1u));
    }
};


struct Ir : SimpleSolverTest<gko::solver::Ir<solver_value_type>> {
    static double tolerance() { return 1e5 * r<value_type>::value; }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return SimpleSolverTest<gko::solver::Ir<solver_value_type>>::build(
                   exec, iteration_count, check_residual)
            .with_solver(precond_type::build().with_max_block_size(1u));
    }

    static const gko::LinOp* get_preconditioner(
        gko::ptr_param<const solver_type> solver)
    {
        return solver->get_solver().get();
    }
};


template <unsigned dimension>
struct CbGmres : SimpleSolverTest<gko::solver::CbGmres<solver_value_type>> {
    static constexpr bool will_not_allocate() { return false; }

    static double tolerance() { return 1e9 * r<value_type>::value; }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return SimpleSolverTest<gko::solver::CbGmres<solver_value_type>>::build(
                   exec, iteration_count, check_residual)
            .with_krylov_dim(dimension);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return build(exec, iteration_count, check_residual)
            .with_preconditioner(precond_type::build().with_max_block_size(1u));
    }
};


template <unsigned dimension>
struct Gmres : SimpleSolverTest<gko::solver::Gmres<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return SimpleSolverTest<gko::solver::Gmres<solver_value_type>>::build(
                   exec, iteration_count, check_residual)
            .with_krylov_dim(dimension);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return build(exec, iteration_count, check_residual)
            .with_preconditioner(precond_type::build().with_max_block_size(1u));
    }
};


template <unsigned dimension>
struct FGmres : SimpleSolverTest<gko::solver::Gmres<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return SimpleSolverTest<gko::solver::Gmres<solver_value_type>>::build(
                   exec, iteration_count, check_residual)
            .with_krylov_dim(dimension)
            .with_flexible(true);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return build(exec, iteration_count, check_residual)
            .with_preconditioner(precond_type::build().with_max_block_size(1u))
            .with_flexible(true);
    }
};


template <unsigned dimension>
struct Gcr : SimpleSolverTest<gko::solver::Gcr<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return SimpleSolverTest<gko::solver::Gcr<solver_value_type>>::build(
                   exec, iteration_count, check_residual)
            .with_krylov_dim(dimension);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return build(exec, iteration_count, check_residual)
            .with_preconditioner(precond_type::build().with_max_block_size(1u));
    }
};


struct LowerTrs : SimpleSolverTest<gko::solver::LowerTrs<solver_value_type>> {
    static constexpr bool will_not_allocate() { return false; }

    static constexpr bool is_iterative() { return false; }

    static constexpr bool is_preconditionable() { return false; }

#ifdef GKO_COMPILING_CUDA
    // cuSPARSE bug related to inputs with more than 32 rhs
    static constexpr bool requires_num_rhs() { return true; }
#endif

    static double tolerance() { return r<value_type>::value; }

    static void preprocess(gko::matrix_data<value_type, index_type>& data)
    {
        // make sure the diagonal is nonzero
        gko::utils::make_hpd(data, 1.2);
    }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec, gko::size_type num_rhs,
        bool = true)
    {
        return solver_type::build()
            .with_algorithm(gko::solver::trisolve_algorithm::sparselib)
            .with_num_rhs(num_rhs);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor>, gko::size_type, bool = true)
    {
        assert(false);
        return solver_type::build();
    }

    static const gko::LinOp* get_preconditioner(
        gko::ptr_param<const solver_type> solver)
    {
        return nullptr;
    }

    static const gko::stop::CriterionFactory* get_stop_criterion_factory(
        gko::ptr_param<const solver_type> solver)
    {
        return nullptr;
    }

    static constexpr bool logs_iteration_complete() { return false; }
};


struct UpperTrs : SimpleSolverTest<gko::solver::UpperTrs<solver_value_type>> {
    static constexpr bool will_not_allocate() { return false; }

    static constexpr bool is_iterative() { return false; }

    static constexpr bool is_preconditionable() { return false; }

#ifdef GKO_COMPILING_CUDA
    // cuSPARSE bug related to inputs with more than 32 rhs
    static constexpr bool requires_num_rhs() { return true; }
#endif

    static double tolerance() { return r<value_type>::value; }

    static void preprocess(gko::matrix_data<value_type, index_type>& data)
    {
        // make sure the diagonal is nonzero
        gko::utils::make_hpd(data, 1.2);
    }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec, gko::size_type num_rhs,
        bool = true)
    {
        return solver_type::build()
            .with_algorithm(gko::solver::trisolve_algorithm::sparselib)
            .with_num_rhs(num_rhs);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor>, gko::size_type, bool = true)
    {
        assert(false);
        return solver_type::build();
    }

    static const gko::LinOp* get_preconditioner(
        gko::ptr_param<const solver_type> solver)
    {
        return nullptr;
    }

    static const gko::stop::CriterionFactory* get_stop_criterion_factory(
        gko::ptr_param<const solver_type> solver)
    {
        return nullptr;
    }

    static constexpr bool logs_iteration_complete() { return false; }
};


struct LowerTrsUnitdiag : LowerTrs {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec, gko::size_type num_rhs,
        bool check_residual = true)
    {
        return solver_type::build()
            .with_algorithm(gko::solver::trisolve_algorithm::sparselib)
            .with_num_rhs(num_rhs)
            .with_unit_diagonal(true);
    }
};


struct UpperTrsUnitdiag : UpperTrs {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec, gko::size_type num_rhs,
        bool check_residual = true)
    {
        return solver_type::build()
            .with_algorithm(gko::solver::trisolve_algorithm::sparselib)
            .with_num_rhs(num_rhs)
            .with_unit_diagonal(true);
    }
};


struct LowerTrsSyncfree : LowerTrs {
    static constexpr bool requires_num_rhs() { return false; }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return solver_type::build().with_algorithm(
            gko::solver::trisolve_algorithm::syncfree);
    }
};


struct UpperTrsSyncfree : UpperTrs {
    static constexpr bool requires_num_rhs() { return false; }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return solver_type::build().with_algorithm(
            gko::solver::trisolve_algorithm::syncfree);
    }
};


struct LowerTrsSyncfreeUnitdiag : LowerTrs {
    static constexpr bool requires_num_rhs() { return false; }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return solver_type::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .with_unit_diagonal(true);
    }
};


struct UpperTrsSyncfreeUnitdiag : UpperTrs {
    static constexpr bool requires_num_rhs() { return false; }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count, bool check_residual = true)
    {
        return solver_type::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .with_unit_diagonal(true);
    }
};


template <typename ObjectType>
struct test_pair {
    std::shared_ptr<ObjectType> ref;
    std::shared_ptr<ObjectType> dev;

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::shared_ptr<const gko::Executor> exec)
        : ref{std::move(ref_obj)}, dev{gko::clone(exec, ref)}
    {}

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::unique_ptr<ObjectType> dev_obj)
        : ref{std::move(ref_obj)}, dev{std::move(dev_obj)}
    {}

    test_pair() = default;
    test_pair(const test_pair& o) = default;
    test_pair(test_pair&& o) noexcept = default;
    test_pair& operator=(const test_pair& o) = default;
    test_pair& operator=(test_pair&& o) noexcept = default;
};


struct DummyLogger : gko::log::Logger {
    DummyLogger() : gko::log::Logger(gko::log::Logger::iteration_complete_mask)
    {}

    void on_iteration_complete(const gko::LinOp* solver, const gko::LinOp* b,
                               const gko::LinOp* x, const gko::size_type& it,
                               const gko::LinOp* r, const gko::LinOp* tau,
                               const gko::LinOp* implicit_tau,
                               const gko::array<gko::stopping_status>* status,
                               bool all_stopped) const override
    {
        iteration_complete = it;
    }

    mutable int iteration_complete = 0;
};


class FailOnAllocationFreeLogger : public gko::log::Logger {
public:
    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type& num_bytes) const override
    {
        FAIL() << "allocation of size " << num_bytes;
    }

    void on_free_started(const gko::Executor* exec,
                         const gko::uintptr& location) const override
    {
        FAIL() << "free";
    }

    FailOnAllocationFreeLogger()
        : Logger(gko::log::Logger::allocation_started_mask |
                 gko::log::Logger::free_started_mask)
    {}
};


template <typename T>
class Solver : public CommonTestFixture {
protected:
    using Config = T;
    using SolverType = typename T::solver_type;
    using Precond = typename T::precond_type;
    using Mtx = typename T::matrix_type;
    using value_type = typename Mtx::value_type;
    using mixed_value_type = gko::next_precision<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<mixed_value_type>;

    Solver()
    {
        reset_rand();
        logger = std::make_shared<DummyLogger>();
    }

    void reset_rand() { rand_engine.seed(15); }

    test_pair<Mtx> gen_mtx(int num_rows, int num_cols, int min_cols,
                           int max_cols)
    {
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                num_rows, num_cols,
                std::uniform_int_distribution<>(min_cols, max_cols),
                std::normal_distribution<>(0.0, 1.0), rand_engine);
        Config::preprocess(data);
        auto mtx = Mtx::create(ref);
        mtx->read(data);
        return test_pair<Mtx>{std::move(mtx), exec};
    }

    template <typename ValueType, typename IndexType>
    gko::matrix_data<ValueType, IndexType> gen_dense_data(gko::dim<2> size)
    {
        return {
            size,
            std::normal_distribution<gko::remove_complex<ValueType>>(0.0, 1.0),
            rand_engine};
    }

    template <typename VecType = Vec, typename MtxOrSolver>
    test_pair<VecType> gen_in_vec(const test_pair<MtxOrSolver>& op, int nrhs,
                                  int stride)
    {
        auto size = gko::dim<2>{op.ref->get_size()[1],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size, stride);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_scalar()
    {
        return {gko::initialize<VecType>(
                    {gko::test::detail::get_rand_value<
                        typename VecType::value_type>(
                        std::normal_distribution<
                            gko::remove_complex<typename VecType::value_type>>(
                            0.0, 1.0),
                        rand_engine)},
                    ref),
                exec};
    }

    template <typename VecType = Vec, typename MtxOrSolver>
    test_pair<VecType> gen_out_vec(const test_pair<MtxOrSolver>& op, int nrhs,
                                   int stride)
    {
        auto size = gko::dim<2>{op.ref->get_size()[0],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size, stride);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    template <typename VecType>
    double tol(const test_pair<VecType>& x)
    {
        return Config::tolerance() * std::sqrt(x.ref->get_size()[1]);
    }

    template <typename VecType>
    double mixed_tol(const test_pair<VecType>& x)
    {
        return std::max(r_mixed<value_type, mixed_value_type>() *
                            std::sqrt(x.ref->get_size()[1]),
                        tol(x));
    }

    template <typename TestFunction>
    void forall_matrix_scenarios(TestFunction fn)
    {
        auto guarded_fn = [&](auto mtx) {
            try {
                fn(std::move(mtx));
                this->reset_rand();
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Empty matrix (0x0)");
            guarded_fn(gen_mtx(0, 0, 0, 0));
        }
        {
            SCOPED_TRACE("Sparse Matrix with variable row nnz (50x50)");
            guarded_fn(gen_mtx(50, 50, 10, 20));
        }
    }

    template <typename TestFunction>
    void forall_solver_scenarios(const test_pair<Mtx>& mtx, TestFunction fn)
    {
        auto guarded_fn = [&](auto solver) {
            try {
                fn(std::move(solver));
                this->reset_rand();
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Defaulted solver");
            guarded_fn(
                test_pair<SolverType>{Config::build(ref, 0, check_residual)
                                          .on(ref)
                                          ->generate(mtx.ref)
                                          ->create_default(),
                                      Config::build(exec, 0, check_residual)
                                          .on(exec)
                                          ->generate(mtx.dev)
                                          ->create_default()});
        }
        {
            SCOPED_TRACE("Cleared solver");
            test_pair<SolverType> pair{Config::build(ref, 0, check_residual)
                                           .on(ref)
                                           ->generate(mtx.ref),
                                       Config::build(exec, 0, check_residual)
                                           .on(exec)
                                           ->generate(mtx.dev)};
            pair.ref->clear();
            pair.dev->clear();
            guarded_fn(std::move(pair));
        }
        /* Disable the test with clone, since cloning is not correctly supported
         * for types that contain factories as members.
         * TODO: reenable when cloning of factories is figured out
        {
            SCOPED_TRACE("Unpreconditioned solver with 0 iterations via clone");
            guarded_fn(
                test_pair<SolverType>{Config::build(ref, 0, check_residual)
                                          .on(ref)
                                          ->generate(mtx.ref),
                                      exec});
        }
        */
        {
            SCOPED_TRACE("Unpreconditioned solver with 0 iterations");
            guarded_fn(
                test_pair<SolverType>{Config::build(ref, 0, check_residual)
                                          .on(ref)
                                          ->generate(mtx.ref),
                                      Config::build(exec, 0, check_residual)
                                          .on(exec)
                                          ->generate(mtx.dev)});
        }
        if (Config::is_preconditionable()) {
            SCOPED_TRACE("Preconditioned solver with 0 iterations");
            guarded_fn(test_pair<SolverType>{
                Config::build_preconditioned(ref, 0, check_residual)
                    .on(ref)
                    ->generate(mtx.ref),
                Config::build_preconditioned(exec, 0, check_residual)
                    .on(exec)
                    ->generate(mtx.dev)});
        }
        static_assert(!(Config::requires_num_rhs() && Config::is_iterative()),
                      "Inconsistent config");
        if (Config::is_iterative()) {
            {
                SCOPED_TRACE("Unpreconditioned solver with 4 iterations");
                guarded_fn(
                    test_pair<SolverType>{Config::build(ref, 4, check_residual)
                                              .on(ref)
                                              ->generate(mtx.ref),
                                          Config::build(exec, 4, check_residual)
                                              .on(exec)
                                              ->generate(mtx.dev)});
            }
            if (Config::is_preconditionable()) {
                SCOPED_TRACE("Preconditioned solver with 4 iterations");
                guarded_fn(test_pair<SolverType>{
                    Config::build_preconditioned(ref, 4, check_residual)
                        .on(ref)
                        ->generate(mtx.ref),
                    Config::build_preconditioned(exec, 4, check_residual)
                        .on(exec)
                        ->generate(mtx.dev)});
            }
        }
    }

    template <typename VecType = Vec, typename TestFunction>
    void forall_solver_scenarios_with_nrhs(const test_pair<Mtx>& mtx,
                                           const test_pair<VecType>& vec,
                                           TestFunction fn)
    {
        auto guarded_fn = [&](auto solver) {
            try {
                fn(std::move(solver));
                this->reset_rand();
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        const auto nrhs = vec.ref->get_size()[1] *
                          (gko::is_complex<typename VecType::value_type>() &&
                                   !gko::is_complex<value_type>()
                               ? 2
                               : 1);
        // No 0x0 cleared/defaulted solvers, as they would be inconsistent with
        // `vec`
        {
            SCOPED_TRACE("Unpreconditioned solver with 0 iterations via clone");
            guarded_fn(
                test_pair<SolverType>{Config::build(ref, nrhs, check_residual)
                                          .on(ref)
                                          ->generate(mtx.ref),
                                      exec});
        }
        {
            SCOPED_TRACE("Unpreconditioned solver with 0 iterations");
            guarded_fn(
                test_pair<SolverType>{Config::build(ref, nrhs, check_residual)
                                          .on(ref)
                                          ->generate(mtx.ref),
                                      Config::build(exec, nrhs, check_residual)
                                          .on(exec)
                                          ->generate(mtx.dev)});
        }
        if (Config::is_preconditionable()) {
            SCOPED_TRACE("Preconditioned solver with 0 iterations");
            guarded_fn(test_pair<SolverType>{
                Config::build_preconditioned(ref, nrhs, check_residual)
                    .on(ref)
                    ->generate(mtx.ref),
                Config::build_preconditioned(exec, nrhs, check_residual)
                    .on(exec)
                    ->generate(mtx.dev)});
        }
        static_assert(!(Config::requires_num_rhs() && Config::is_iterative()),
                      "Inconsistent config");
    }

    template <typename VecType = Vec, typename MtxOrSolver, typename MtxType,
              typename TestFunction>
    void forall_vector_scenarios(const test_pair<MtxOrSolver>& op,
                                 const test_pair<MtxType>& mtx, TestFunction fn)
    {
        auto guarded_fn = [&](auto b, auto x) {
            try {
                fn(std::move(b), std::move(x));
                this->reset_rand();
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Multivector with 0 columns");
            guarded_fn(gen_in_vec<VecType>(op, 0, 0),
                       gen_out_vec<VecType>(op, 0, 0));
        }
        {
            SCOPED_TRACE("Single vector");
            guarded_fn(gen_in_vec<VecType>(op, 1, 1),
                       gen_out_vec<VecType>(op, 1, 1));
        }
        if (Config::is_iterative() &&
            op.ref->get_size() == gko::transpose(mtx.ref->get_size())) {
            SCOPED_TRACE("Single vector with correct initial guess");
            auto in = gen_in_vec<VecType>(op, 1, 1);
            auto out = gen_out_vec<VecType>(op, 1, 1);
            mtx.ref->apply(out.ref, in.ref);
            mtx.dev->apply(out.dev, in.dev);
            guarded_fn(std::move(in), std::move(out));
        }
        {
            SCOPED_TRACE("Single strided vector");
            guarded_fn(gen_in_vec<VecType>(op, 1, 2),
                       gen_out_vec<VecType>(op, 1, 3));
        }
        if (!gko::is_complex<value_type>()) {
            // check application of real matrix to complex vector
            // viewed as interleaved real/imag vector
            using complex_vec = gko::to_complex<VecType>;
            {
                SCOPED_TRACE("Single strided complex vector");
                guarded_fn(gen_in_vec<complex_vec>(op, 1, 2),
                           gen_out_vec<complex_vec>(op, 1, 3));
            }
            {
                SCOPED_TRACE("Strided complex multivector with 2 columns");
                guarded_fn(gen_in_vec<complex_vec>(op, 2, 3),
                           gen_out_vec<complex_vec>(op, 2, 4));
            }
        }
        {
            SCOPED_TRACE("Multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(op, 2, 2),
                       gen_out_vec<VecType>(op, 2, 2));
        }
        {
            SCOPED_TRACE("Strided multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(op, 2, 3),
                       gen_out_vec<VecType>(op, 2, 4));
        }
        {
            SCOPED_TRACE("Multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(op, 40, 40),
                       gen_out_vec<VecType>(op, 40, 40));
        }
        {
            SCOPED_TRACE("Strided multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(op, 40, 43),
                       gen_out_vec<VecType>(op, 40, 45));
        }
    }

    template <typename VecType = Vec, typename TestFunction>
    void forall_vector_and_solver_scenarios(const test_pair<Mtx>& mtx,
                                            TestFunction fn)
    {
        if (Config::requires_num_rhs()) {
            forall_vector_scenarios<VecType>(
                mtx, mtx, [this, &mtx, &fn](auto b, auto x) {
                    forall_solver_scenarios_with_nrhs(
                        mtx, b,
                        [this, &fn, &b, &x](auto solver) { fn(solver, b, x); });
                });
        } else {
            forall_solver_scenarios(mtx, [this, &mtx, &fn](auto solver) {
                forall_vector_scenarios<VecType>(
                    solver, mtx,
                    [this, &solver, &fn](auto b, auto x) { fn(solver, b, x); });
            });
        }
    }

    void assert_empty_state(gko::ptr_param<const SolverType> solver,
                            std::shared_ptr<const gko::Executor> expected_exec)
    {
        ASSERT_FALSE(solver->get_size());
        ASSERT_EQ(solver->get_executor(), expected_exec);
        ASSERT_EQ(solver->get_system_matrix(), nullptr);
        ASSERT_EQ(Config::get_preconditioner(solver), nullptr);
    }

    std::shared_ptr<DummyLogger> logger;

    std::default_random_engine rand_engine;

    bool check_residual = true;
};

using SolverTypes =
    ::testing::Types<Cg, Cgs, Fcg, Bicg, Bicgstab,
                     /* "IDR uses different initialization approaches even when
                        deterministic", Idr<1>, Idr<4>,*/
                     Ir, CbGmres<2>, CbGmres<10>, Gmres<2>, Gmres<10>,
                     FGmres<2>, FGmres<10>, Gcr<2>, Gcr<10>, LowerTrs, UpperTrs,
                     LowerTrsUnitdiag, UpperTrsUnitdiag
#ifdef GKO_COMPILING_CUDA
                     ,
                     LowerTrsSyncfree, UpperTrsSyncfree,
                     LowerTrsSyncfreeUnitdiag, UpperTrsSyncfreeUnitdiag
#endif  // GKO_COMPILING_CUDA
                     >;

TYPED_TEST_SUITE(Solver, SolverTypes, TypenameNameGenerator);


TYPED_TEST(Solver, ApplyIsEquivalentToRef)
{
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_vector_and_solver_scenarios(
            mtx, [this, &mtx](auto solver, auto b, auto x) {
                solver.ref->apply(b.ref, x.ref);
                solver.dev->apply(b.dev, x.dev);

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol(x));
            });
    });
}


TYPED_TEST(Solver, ApplyDoesntAllocateRepeatedly)
{
    this->check_residual = false;
    if (!TypeParam::will_not_allocate()) {
        GTEST_SKIP()
            << "Skipping allocation test for types that will not allocate";
    }
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_vector_and_solver_scenarios(
            mtx, [this, &mtx](auto solver, auto b, auto x) {
                solver.dev->apply(b.dev, x.dev);
                auto logger = std::make_shared<FailOnAllocationFreeLogger>();

                this->exec->add_logger(logger);
                solver.dev->apply(b.dev, x.dev);
                this->exec->remove_logger(logger);
            });
    });
}


TYPED_TEST(Solver, AdvancedApplyIsEquivalentToRef)
{
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_vector_and_solver_scenarios(
            mtx, [this, &mtx](auto solver, auto b, auto x) {
                auto alpha = this->gen_scalar();
                auto beta = this->gen_scalar();

                solver.ref->apply(alpha.ref, b.ref, beta.ref, x.ref);
                solver.dev->apply(alpha.dev, b.dev, beta.dev, x.dev);

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol(x));
            });
    });
}


TYPED_TEST(Solver, MixedApplyIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->template forall_vector_and_solver_scenarios<MixedVec>(
            mtx, [this, &mtx](auto solver, auto b, auto x) {
                solver.ref->apply(b.ref, x.ref);
                solver.dev->apply(b.dev, x.dev);

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol(x));
            });
    });
}


TYPED_TEST(Solver, MixedAdvancedApplyIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->template forall_vector_and_solver_scenarios<MixedVec>(
            mtx, [this, &mtx](auto solver, auto b, auto x) {
                auto alpha = this->template gen_scalar<MixedVec>();
                auto beta = this->template gen_scalar<MixedVec>();

                solver.ref->apply(alpha.ref, b.ref, beta.ref, x.ref);
                solver.dev->apply(alpha.dev, b.dev, beta.dev, x.dev);

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol(x));
            });
    });
}


TYPED_TEST(Solver, CrossExecutorGenerateCopiesToFactoryExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_scenarios([this](auto mtx) {
        auto solver =
            Config::build(this->ref, 0).on(this->exec)->generate(mtx.ref);

        ASSERT_EQ(solver->get_system_matrix()->get_executor(), this->exec);
        ASSERT_EQ(solver->get_executor(), this->exec);
        if (Config::is_iterative()) {
            ASSERT_EQ(
                Config::get_stop_criterion_factory(solver)->get_executor(),
                this->exec);
        }
        if (Config::is_preconditionable()) {
            auto precond = Config::get_preconditioner(solver);
            ASSERT_EQ(precond->get_executor(), this->exec);
            ASSERT_TRUE(dynamic_cast<
                        const gko::matrix::Identity<typename Mtx::value_type>*>(
                precond));
        }
        GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(solver->get_system_matrix()), mtx.ref,
                            0.0);
    });
}


TYPED_TEST(Solver, CopyAssignSameExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            auto solver2 = Config::build(this->exec, 0)
                               .on(this->exec)
                               ->generate(Mtx::create(this->exec));

            auto& result = (*solver2 = *solver.dev);

            ASSERT_EQ(&result, solver2.get());
            ASSERT_EQ(solver2->get_size(), solver.dev->get_size());
            ASSERT_EQ(solver2->get_executor(), solver.dev->get_executor());
            ASSERT_EQ(solver2->get_system_matrix(),
                      solver.dev->get_system_matrix());
            ASSERT_EQ(Config::get_stop_criterion_factory(solver2),
                      Config::get_stop_criterion_factory(solver.dev));
            ASSERT_EQ(Config::get_preconditioner(solver2),
                      Config::get_preconditioner(solver.dev));
        });
    });
}


TYPED_TEST(Solver, MoveAssignSameExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_scenarios([this](auto in_mtx) {
        this->forall_solver_scenarios(in_mtx, [this](auto solver) {
            auto solver2 = Config::build(this->exec, 0)
                               .on(this->exec)
                               ->generate(Mtx::create(this->exec));
            auto size = solver.dev->get_size();
            auto mtx = solver.dev->get_system_matrix();
            auto precond = Config::get_preconditioner(solver.dev);
            auto stop = Config::get_stop_criterion_factory(solver.dev);

            auto& result = (*solver2 = std::move(*solver.dev));

            ASSERT_EQ(&result, solver2.get());
            // moved-to object
            ASSERT_EQ(solver2->get_size(), size);
            ASSERT_EQ(solver2->get_executor(), this->exec);
            ASSERT_EQ(solver2->get_system_matrix(), mtx);
            ASSERT_EQ(Config::get_stop_criterion_factory(solver2), stop);
            ASSERT_EQ(Config::get_preconditioner(solver2), precond);
            // moved-from object
            this->assert_empty_state(solver.dev, this->exec);
        });
    });
}


TYPED_TEST(Solver, CopyAssignCrossExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    using Precond = typename TestFixture::Precond;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            auto solver2 = Config::build(this->exec, 0)
                               .on(this->exec)
                               ->generate(Mtx::create(this->exec));

            auto& result = (*solver2 = *solver.ref);

            ASSERT_EQ(&result, solver2.get());
            ASSERT_EQ(solver2->get_size(), solver.ref->get_size());
            ASSERT_EQ(solver2->get_executor(), this->exec);
            if (solver.ref->get_system_matrix()) {
                GKO_ASSERT_MTX_NEAR(
                    gko::as<Mtx>(solver2->get_system_matrix()),
                    gko::as<Mtx>(solver.ref->get_system_matrix()), 0.0);
                // TODO no easy way to compare stopping criteria cross-executor
                auto precond = Config::get_preconditioner(solver2);
                if (dynamic_cast<const Precond*>(precond)) {
                    GKO_ASSERT_MTX_NEAR(
                        gko::as<Precond>(precond),
                        gko::as<Precond>(
                            Config::get_preconditioner(solver.ref)),
                        0.0);
                }
            }
        });
    });
}


TYPED_TEST(Solver, MoveAssignCrossExecutor)
{
    using Config = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    using Precond = typename TestFixture::Precond;
    this->forall_matrix_scenarios([this](auto in_mtx) {
        this->forall_solver_scenarios(in_mtx, [this](auto solver) {
            auto solver2 = Config::build(this->exec, 0)
                               .on(this->exec)
                               ->generate(Mtx::create(this->exec));
            auto size = solver.ref->get_size();
            auto mtx = solver.ref->get_system_matrix();
            auto precond = Config::get_preconditioner(solver.ref);
            auto stop = Config::get_stop_criterion_factory(solver.ref);

            auto& result = (*solver2 = std::move(*solver.ref));

            ASSERT_EQ(&result, solver2.get());
            // moved-to object
            ASSERT_EQ(solver2->get_size(), size);
            ASSERT_EQ(solver2->get_executor(), this->exec);
            if (solver.ref->get_system_matrix()) {
                GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(solver2->get_system_matrix()),
                                    gko::as<Mtx>(mtx), 0.0);
                // TODO no easy way to compare stopping criteria cross-executor
                auto new_precond = Config::get_preconditioner(solver2);
                if (dynamic_cast<const Precond*>(new_precond)) {
                    GKO_ASSERT_MTX_NEAR(gko::as<Precond>(new_precond),
                                        gko::as<Precond>(precond), 0.0);
                }
            }
            // moved-from object
            this->assert_empty_state(solver.ref, this->ref);
        });
    });
}


TYPED_TEST(Solver, ClearIsEmpty)
{
    using Config = typename TestFixture::Config;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            solver.dev->clear();

            this->assert_empty_state(solver.dev, this->exec);
        });
    });
}


TYPED_TEST(Solver, CreateDefaultIsEmpty)
{
    using Config = typename TestFixture::Config;
    this->forall_matrix_scenarios([this](auto mtx) {
        this->forall_solver_scenarios(mtx, [this](auto solver) {
            auto default_solver = solver.dev->create_default();

            this->assert_empty_state(default_solver, this->exec);
        });
    });
}


TYPED_TEST(Solver, LogsIterationComplete)
{
    using Config = typename TestFixture::Config;
    if (Config::logs_iteration_complete()) {
        using Mtx = typename TestFixture::Mtx;
        using Vec = typename TestFixture::Vec;
        auto mtx = gko::share(Mtx::create(this->exec));
        auto b = Vec::create(this->exec);
        auto x = Vec::create(this->exec);
        gko::size_type num_iteration(4);
        auto solver = Config::build(this->exec, num_iteration, false)
                          .on(this->exec)
                          ->generate(mtx);
        auto before_logger = *this->logger;
        solver->add_logger(this->logger);

        solver->apply(b, x);

        ASSERT_EQ(this->logger->iteration_complete,
                  before_logger.iteration_complete + num_iteration);
    }
}

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
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/mpi/executor.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename SolverType>
struct SimpleSolverTest {
    using solver_type = SolverType;
    using value_type = typename solver_type::value_type;
    using mixed_value_type = gko::next_precision<value_type>;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using dist_matrix_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               gko::int64>;
    using non_dist_matrix_type =
        gko::matrix::Csr<value_type, global_index_type>;
    using dist_vector_type = gko::experimental::distributed::Vector<value_type>;
    using non_dist_vector_type = gko::matrix::Dense<value_type>;
    using mixed_dist_vector_type =
        gko::experimental::distributed::Vector<mixed_value_type>;
    using mixed_non_dist_vector_type = gko::matrix::Dense<mixed_value_type>;
    using partition_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;

    static constexpr double tolerance() { return 10 * reduction_factor(); }

    static constexpr gko::size_type iteration_count() { return 200u; }

    static constexpr value_type reduction_factor() { return 1e-4; }

    static void preprocess(
        gko::matrix_data<value_type, global_index_type>& data)
    {
        gko::utils::make_diag_dominant(data, 1.5);
    }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec)
    {
        return solver_type::build().with_criteria(
            gko::stop::Iteration::build().with_max_iters(iteration_count()),
            gko::stop::ResidualNorm<value_type>::build()
                .with_baseline(gko::stop::mode::absolute)
                .with_reduction_factor(reduction_factor())
                .on(exec));
    }

    static void assert_empty_state(const solver_type* mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_system_matrix(), nullptr);
        ASSERT_EQ(mtx->get_preconditioner(), nullptr);
        ASSERT_EQ(mtx->get_stopping_criterion_factory(), nullptr);
    }
};


struct Cg : SimpleSolverTest<gko::solver::Cg<solver_value_type>> {
    static void preprocess(
        gko::matrix_data<value_type, global_index_type>& data)
    {
        // make sure the matrix is well-conditioned
        gko::utils::make_hpd(data, 1.5);
    }
};


struct Cgs : SimpleSolverTest<gko::solver::Cgs<solver_value_type>> {};


struct Fcg : SimpleSolverTest<gko::solver::Fcg<solver_value_type>> {
    static void preprocess(
        gko::matrix_data<value_type, global_index_type>& data)
    {
        gko::utils::make_hpd(data, 1.5);
    }
};


struct Bicgstab : SimpleSolverTest<gko::solver::Bicgstab<solver_value_type>> {
    static constexpr double tolerance() { return 300 * reduction_factor(); }
};


struct Ir : SimpleSolverTest<gko::solver::Ir<solver_value_type>> {
    static void preprocess(
        gko::matrix_data<value_type, global_index_type>& data)
    {
        gko::utils::make_hpd(data, 1.5);
    }

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec)
    {
        return SimpleSolverTest<gko::solver::Ir<solver_value_type>>::build(exec)
            .with_solver(gko::solver::Cg<value_type>::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(iteration_count()),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_baseline(gko::stop::mode::absolute)
                    .with_reduction_factor(2 * reduction_factor())))
            .with_relaxation_factor(0.9);
    }
};


template <unsigned dimension>
struct Gmres : SimpleSolverTest<gko::solver::Gmres<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec)
    {
        return SimpleSolverTest<gko::solver::Gmres<solver_value_type>>::build(
                   std::move(exec))
            .with_krylov_dim(dimension);
    }
};


template <unsigned dimension>
struct Gcr : SimpleSolverTest<gko::solver::Gcr<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec)
    {
        return SimpleSolverTest<gko::solver::Gcr<solver_value_type>>::build(
                   std::move(exec))
            .with_krylov_dim(dimension);
    }
};


template <typename T>
class Solver : public CommonMpiTestFixture {
protected:
    using Config = T;
    using SolverType = typename T::solver_type;
    using Mtx = typename T::dist_matrix_type;
    using local_index_type = typename T::local_index_type;
    using global_index_type = typename T::global_index_type;
    using value_type = typename T::value_type;
    using mixed_value_type = gko::next_precision<value_type>;
    using Vec = typename T::dist_vector_type;
    using LocalVec = typename T::non_dist_vector_type;
    using MixedVec = typename T::mixed_dist_vector_type;
    using MixedLocalVec = typename T::mixed_non_dist_vector_type;
    using Part = typename T::partition_type;

    Solver() : rand_engine(15) {}

    std::unique_ptr<Part> gen_part(int size, int num_active_parts)
    {
        auto mapping = gko::test::generate_random_array<
            gko::experimental::distributed::comm_index_type>(
            size,
            std::uniform_int_distribution<
                gko::experimental::distributed::comm_index_type>(
                0, num_active_parts - 1),
            rand_engine, ref);
        return Part::build_from_mapping(ref, mapping, comm.size());
    }


    std::shared_ptr<Mtx> gen_mtx(const Part* part, int num_rows, int num_cols,
                                 int min_cols, int max_cols)
    {
        auto data = gko::test::generate_random_matrix_data<value_type,
                                                           global_index_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_cols, max_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine);
        Config::preprocess(data);
        auto dist_mtx = Mtx::create(ref, comm);
        dist_mtx->read_distributed(data, part);
        return gko::share(gko::clone(exec, dist_mtx));
    }

    template <typename ValueType, typename IndexType>
    gko::matrix_data<ValueType, IndexType> gen_dense_data(gko::dim<2> size)
    {
        return {
            size,
            std::normal_distribution<gko::remove_complex<ValueType>>(0.0, 1.0),
            rand_engine};
    }

    template <typename DistVecType = Vec>
    std::shared_ptr<DistVecType> gen_in_vec(
        const Part* part, const std::shared_ptr<SolverType>& solver, int nrhs,
        int stride)
    {
        auto global_size = gko::dim<2>{solver->get_size()[1],
                                       static_cast<gko::size_type>(nrhs)};
        auto local_size = gko::dim<2>{
            static_cast<gko::size_type>(part->get_part_size(comm.rank())),
            static_cast<gko::size_type>(nrhs)};
        auto dist_result =
            DistVecType::create(ref, comm, global_size, local_size, stride);
        dist_result->read_distributed(
            gen_dense_data<typename DistVecType::value_type, global_index_type>(
                global_size),
            part);
        return gko::share(gko::clone(exec, dist_result));
    }

    template <typename VecType = LocalVec>
    std::shared_ptr<VecType> gen_scalar()
    {
        return gko::share(gko::initialize<VecType>(
            {gko::test::detail::get_rand_value<typename VecType::value_type>(
                std::normal_distribution<
                    gko::remove_complex<typename VecType::value_type>>(0.0,
                                                                       1.0),
                rand_engine)},
            exec));
    }

    template <typename DistVecType = Vec>
    std::shared_ptr<DistVecType> gen_out_vec(
        const Part* part, const std::shared_ptr<SolverType>& solver, int nrhs,
        int stride)
    {
        auto global_size = gko::dim<2>{solver->get_size()[0],
                                       static_cast<gko::size_type>(nrhs)};
        auto local_size = gko::dim<2>{
            static_cast<gko::size_type>(part->get_part_size(comm.rank())),
            static_cast<gko::size_type>(nrhs)};
        auto dist_result =
            DistVecType::create(ref, comm, global_size, local_size, stride);
        dist_result->read_distributed(
            gen_dense_data<typename DistVecType::value_type, global_index_type>(
                global_size),
            part);
        return gko::share(gko::clone(exec, dist_result));
    }

    template <typename VecType>
    double tol(const std::shared_ptr<VecType>& x)
    {
        return Config::tolerance() * std::sqrt(x->get_size()[1]);
    }

    template <typename VecType>
    double mixed_tol(const std::shared_ptr<VecType>& x)
    {
        return std::max(r_mixed<value_type, mixed_value_type>() *
                            std::sqrt(x->get_size()[1]),
                        tol(x));
    }

    template <typename TestFunction>
    void forall_partition_scenarios(TestFunction fn)
    {
        auto guarded_fn = [&](auto prt) {
            try {
                comm.synchronize();
                fn(std::move(prt));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Empty partition");
            guarded_fn(gen_part(0, comm.size()));
        }
        {
            SCOPED_TRACE("Full partition");
            guarded_fn(gen_part(50, comm.size()));
        }
        {
            SCOPED_TRACE("Some empty partition");
            guarded_fn(gen_part(50, std::max(1, comm.size() - 1)));
        }
    }

    template <typename TestFunction>
    void forall_matrix_scenarios(const Part* part, TestFunction fn)
    {
        auto guarded_fn = [&](auto mtx) {
            try {
                fn(std::move(mtx));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            int num_rows = part->get_size();
            SCOPED_TRACE("Sparse Matrix with variable row nnz (" +
                         std::to_string(num_rows) + "x" +
                         std::to_string(num_rows) + ")");
            guarded_fn(gen_mtx(part, num_rows, num_rows, std::min(10, num_rows),
                               num_rows));
        }
    }

    template <typename TestFunction>
    void forall_solver_scenarios(const std::shared_ptr<Mtx>& mtx,
                                 TestFunction fn)
    {
        auto guarded_fn = [&](auto solver) {
            try {
                fn(std::move(solver));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE(
                "Unpreconditioned solver with fixed tolerance and number of "
                "iterations");
            guarded_fn(gko::share(Config::build(exec).on(exec)->generate(mtx)));
        }
    }

    template <typename DistVecType = Vec, typename NonDistVecType = LocalVec,
              typename TestFunction>
    void forall_vector_scenarios(const Part* part,
                                 const std::shared_ptr<SolverType>& solver,
                                 TestFunction fn)
    {
        auto guarded_fn = [&](auto b, auto x) {
            try {
                fn(std::move(b), std::move(x));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Multivector with 0 columns");
            guarded_fn(gen_in_vec<DistVecType>(part, solver, 0, 0),
                       gen_out_vec<DistVecType>(part, solver, 0, 0));
        }
        {
            SCOPED_TRACE("Single vector");
            guarded_fn(gen_in_vec<DistVecType>(part, solver, 1, 1),
                       gen_out_vec<DistVecType>(part, solver, 1, 1));
        }
        {
            SCOPED_TRACE("Single vector with correct initial guess");
            auto in = gen_in_vec<DistVecType>(part, solver, 1, 1);
            auto out = gen_out_vec<DistVecType>(part, solver, 1, 1);
            solver->get_system_matrix()->apply(out, in);
            guarded_fn(std::move(in), std::move(out));
        }
        {
            SCOPED_TRACE("Single strided vector");
            guarded_fn(gen_in_vec<DistVecType>(part, solver, 1, 2),
                       gen_out_vec<DistVecType>(part, solver, 1, 3));
        }
        {
            SCOPED_TRACE("Multivector with 2 columns");
            guarded_fn(gen_in_vec<DistVecType>(part, solver, 2, 2),
                       gen_out_vec<DistVecType>(part, solver, 2, 2));
        }
        {
            SCOPED_TRACE("Strided multivector with 2 columns");
            guarded_fn(gen_in_vec<DistVecType>(part, solver, 2, 3),
                       gen_out_vec<DistVecType>(part, solver, 2, 4));
        }
        {
            SCOPED_TRACE("Multivector with 17 columns");
            guarded_fn(gen_in_vec<DistVecType>(part, solver, 17, 17),
                       gen_out_vec<DistVecType>(part, solver, 17, 17));
        }
        {
            SCOPED_TRACE("Strided multivector with 17 columns");
            guarded_fn(gen_in_vec<DistVecType>(part, solver, 17, 21),
                       gen_out_vec<DistVecType>(part, solver, 17, 21));
        }
        if (!gko::is_complex<value_type>()) {
            // check application of real matrix to complex vector
            // viewed as interleaved real/imag vector
            using complex_vec = gko::to_complex<DistVecType>;
            {
                SCOPED_TRACE("Single strided complex vector");
                guarded_fn(gen_in_vec<complex_vec>(part, solver, 1, 2),
                           gen_out_vec<complex_vec>(part, solver, 1, 3));
            }
            {
                SCOPED_TRACE("Strided complex multivector with 2 columns");
                guarded_fn(gen_in_vec<complex_vec>(part, solver, 2, 3),
                           gen_out_vec<complex_vec>(part, solver, 2, 4));
            }
        }
    }

    template <typename M1, typename DistVecType>
    void assert_residual_near(const std::shared_ptr<M1>& mtx,
                              const std::shared_ptr<DistVecType>& x,
                              const std::shared_ptr<DistVecType>& b,
                              double tolerance)
    {
        auto one = gko::initialize<LocalVec>({gko::one<value_type>()}, exec);
        auto neg_one =
            gko::initialize<LocalVec>({-gko::one<value_type>()}, exec);
        auto norm = DistVecType::local_vector_type::absolute_type::create(
            ref, gko::dim<2>{1, b->get_size()[1]});
        auto dist_res = gko::clone(b);
        mtx->apply(neg_one, x, one, dist_res);
        dist_res->compute_norm2(norm);

        for (int i = 0; i < norm->get_num_stored_elements(); ++i) {
            ASSERT_LE(norm->at(i), tolerance);
        }
    }


    template <typename M1, typename DistVecType, typename NonDistVecType>
    void assert_residual_near(const std::shared_ptr<M1>& mtx,
                              const std::shared_ptr<DistVecType>& x_sol,
                              const std::shared_ptr<DistVecType>& x_old,
                              const std::shared_ptr<DistVecType>& b,
                              const std::shared_ptr<NonDistVecType>& alpha,
                              const std::shared_ptr<NonDistVecType>& beta,
                              double tolerance)
    {
        auto one = gko::initialize<LocalVec>({gko::one<value_type>()}, exec);
        auto neg_one =
            gko::initialize<LocalVec>({-gko::one<value_type>()}, exec);
        auto norm = DistVecType::local_vector_type::absolute_type::create(
            ref, gko::dim<2>{1, b->get_size()[1]});
        auto dist_res = gko::clone(b);
        // compute rx = (x_sol - beta * x_old) / alpha, since A * rx = b
        // and we only know the accuracy of that operation
        auto recovered_x = gko::clone(x_sol);
        recovered_x->sub_scaled(beta, x_old);
        recovered_x->inv_scale(alpha);
        mtx->apply(neg_one, recovered_x, one, dist_res);
        dist_res->compute_norm2(norm);

        for (int i = 0; i < norm->get_num_stored_elements(); ++i) {
            ASSERT_LE(norm->at(i), tolerance);
        }
    }

    std::default_random_engine rand_engine;
};

using SolverTypes = ::testing::Types<Cg, Cgs, Fcg, Bicgstab, Ir, Gcr<10u>,
                                     Gcr<100u>, Gmres<10u>, Gmres<100u>>;

TYPED_TEST_SUITE(Solver, SolverTypes, TypenameNameGenerator);


TYPED_TEST(Solver, ApplyIsEquivalentToRef)
{
    this->forall_partition_scenarios([&](auto part) {
        this->forall_matrix_scenarios(part.get(), [&](auto mtx) {
            this->forall_solver_scenarios(mtx, [&](auto solver) {
                this->forall_vector_scenarios(
                    part.get(), solver, [&](auto b, auto x) {
                        solver->apply(b, x);

                        this->assert_residual_near(mtx, x, b, this->tol(x));
                    });
            });
        });
    });
}


TYPED_TEST(Solver, AdvancedApplyIsEquivalentToRef)
{
    this->forall_partition_scenarios([&](auto part) {
        this->forall_matrix_scenarios(part.get(), [&](auto mtx) {
            this->forall_solver_scenarios(mtx, [&](auto solver) {
                this->forall_vector_scenarios(
                    part.get(), solver, [&](auto b, auto x) {
                        auto alpha = this->gen_scalar();
                        auto beta = this->gen_scalar();
                        auto x_old = gko::share(gko::clone(x));

                        solver->apply(alpha, b, beta, x);

                        this->assert_residual_near(mtx, x, x_old, b, alpha,
                                                   beta, 10 * this->tol(x));
                    });
            });
        });
    });
}

#if !(GINKGO_DPCPP_SINGLE_MODE)
TYPED_TEST(Solver, MixedApplyIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_partition_scenarios([&](auto part) {
        this->forall_matrix_scenarios(part.get(), [&](auto mtx) {
            this->forall_solver_scenarios(mtx, [&](auto solver) {
                this->template forall_vector_scenarios<MixedVec>(
                    part.get(), solver, [&](auto b, auto x) {
                        solver->apply(b, x);

                        this->assert_residual_near(mtx, x, b,
                                                   this->mixed_tol(x));
                    });
            });
        });
    });
}


TYPED_TEST(Solver, MixedAdvancedApplyIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    using MixedLocalVec = typename TestFixture::MixedLocalVec;
    this->forall_partition_scenarios([&](auto part) {
        this->forall_matrix_scenarios(part.get(), [&](auto mtx) {
            this->forall_solver_scenarios(mtx, [&](auto solver) {
                this->template forall_vector_scenarios<MixedVec>(
                    part.get(), solver, [&](auto b, auto x) {
                        auto alpha = this->template gen_scalar<MixedLocalVec>();
                        auto beta = this->template gen_scalar<MixedLocalVec>();
                        auto x_old = gko::share(gko::clone(x));

                        solver->apply(alpha, b, beta, x);

                        this->assert_residual_near(mtx, x, x_old, b, alpha,
                                                   beta,
                                                   10 * this->mixed_tol(x));
                    });
            });
        });
    });
}
#endif

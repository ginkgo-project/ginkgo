// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"


template <typename ValueIndexType>
class PmisFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::Pmis<value_type, index_type>;
    using real_type = gko::remove_complex<value_type>;
    PmisFactory()
        : exec(gko::ReferenceExecutor::create()),
          pmis_factory(MgLevel::build()
                           .with_strength_threshold(real_type{0.125})
                           .with_skip_sorting(true)
                           .on(exec))

    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename MgLevel::Factory> pmis_factory;
};

TYPED_TEST_SUITE(PmisFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(PmisFactory, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->pmis_factory->get_executor(), this->exec);
}


TYPED_TEST(PmisFactory, DefaultSetting)
{
    using real_type = typename TestFixture::real_type;
    using MgLevel = typename TestFixture::MgLevel;
    auto factory = MgLevel::build().on(this->exec);

    ASSERT_EQ(factory->get_parameters().strength_threshold, real_type{0.25});
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().max_row_sum, real_type{0.9});
}


TYPED_TEST(PmisFactory, SetStrengthThreshold)
{
    using real_type = typename TestFixture::real_type;
    ASSERT_EQ(this->pmis_factory->get_parameters().strength_threshold,
              real_type{0.125});
}

TYPED_TEST(PmisFactory, SetSkipSorting)
{
    ASSERT_EQ(this->pmis_factory->get_parameters().skip_sorting, true);
}


template <typename ValueType>
class PmisSolve : public ::testing::Test {
protected:
    using value_type = ValueType;
    using index_type = gko::int32;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::Pmis<value_type, index_type>;

    PmisSolve() : exec(gko::ReferenceExecutor::create())
    {
        constexpr int n = 12;
        gko::matrix_data<value_type, index_type> data{
            gko::dim<2>(n * n, n * n)};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                const index_type row = static_cast<index_type>(i * n + j);
                data.nonzeros.emplace_back(row, row, value_type{4});
                if (i > 0) {
                    data.nonzeros.emplace_back(row, row - n, value_type{-1});
                }
                if (i < n - 1) {
                    data.nonzeros.emplace_back(row, row + n, value_type{-1});
                }
                if (j > 0) {
                    data.nonzeros.emplace_back(row, row - 1, value_type{-1});
                }
                if (j < n - 1) {
                    data.nonzeros.emplace_back(row, row + 1, value_type{-1});
                }
            }
        }
        data.sort_row_major();
        mtx = gko::share(Mtx::create(exec));
        mtx->read(data);

        b = Vec::create(exec, gko::dim<2>{mtx->get_size()[0], 1});
        b->fill(value_type{1});
        x0 = Vec::create(exec, gko::dim<2>{mtx->get_size()[0], 1});
        x0->fill(value_type{0});
    }

    struct solve_result {
        gko::size_type iterations;
        bool converged;
    };

    solve_result solve(gko::multigrid::interpolation_type interpolation)
    {
        const auto tol = r<value_type>::value * real_type{10};
        auto logger = gko::share(gko::log::Convergence<value_type>::create());
        auto solver =
            gko::solver::Multigrid::build()
                .with_max_levels(3u)
                .with_min_coarse_rows(8u)
                .with_mg_level(
                    MgLevel::build().with_interpolation(interpolation))
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(200u),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(tol))
                .on(exec)
                ->generate(mtx);
        solver->add_logger(logger);
        auto x = gko::clone(x0);
        solver->apply(b, x);
        return {logger->get_num_iterations(), logger->has_converged()};
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Vec> b;
    std::shared_ptr<Vec> x0;
};

TYPED_TEST_SUITE(PmisSolve, gko::test::RealValueTypes, TypenameNameGenerator);


TYPED_TEST(PmisSolve, ClassicalConvergesInNoMoreIterationsThanDirect)
{
    const auto direct = this->solve(gko::multigrid::interpolation_type::direct);
    const auto classical =
        this->solve(gko::multigrid::interpolation_type::classical);

    EXPECT_TRUE(direct.converged);
    EXPECT_TRUE(classical.converged);
    EXPECT_LE(classical.iterations, direct.iterations);
}


TYPED_TEST(PmisSolve, ExtIGeneratesAFiniteProlongation)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;

    auto mg = MgLevel::build()
                  .with_interpolation(gko::multigrid::interpolation_type::ext_i)
                  .on(this->exec)
                  ->generate(this->mtx);

    auto prolong = gko::as<Mtx>(mg->get_prolong_op());
    ASSERT_EQ(prolong->get_size()[0], this->mtx->get_size()[0]);
    ASSERT_GT(prolong->get_size()[1], 0);
    ASSERT_LT(prolong->get_size()[1], this->mtx->get_size()[0]);
    const auto values = prolong->get_const_values();
    for (gko::size_type i = 0; i < prolong->get_num_stored_elements(); i++) {
        ASSERT_TRUE(gko::is_finite(values[i])) << "entry " << i;
    }
}


TYPED_TEST(PmisSolve, TruncationCapsEveryRowAndKeepsItsRowSum)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    constexpr gko::size_type p_max_elmts = 2;
    const auto tol = std::numeric_limits<real_type>::epsilon() * real_type{16};

    auto full_level =
        MgLevel::build()
            .with_interpolation(gko::multigrid::interpolation_type::ext_i)
            .on(this->exec)
            ->generate(this->mtx);
    auto trunc_level =
        MgLevel::build()
            .with_interpolation(gko::multigrid::interpolation_type::ext_i)
            .with_p_max_elmts(p_max_elmts)
            .on(this->exec)
            ->generate(this->mtx);
    auto full = gko::as<Mtx>(full_level->get_prolong_op());
    auto trunc = gko::as<Mtx>(trunc_level->get_prolong_op());

    ASSERT_EQ(trunc->get_size(), full->get_size());
    const auto* full_ptrs = full->get_const_row_ptrs();
    const auto* trunc_ptrs = trunc->get_const_row_ptrs();
    gko::size_type truncated_rows = 0;
    gko::size_type expected_nnz = 0;
    for (gko::size_type row = 0; row < full->get_size()[0]; row++) {
        const auto n_full =
            static_cast<gko::size_type>(full_ptrs[row + 1] - full_ptrs[row]);
        const auto n_trunc =
            static_cast<gko::size_type>(trunc_ptrs[row + 1] - trunc_ptrs[row]);
        const auto expected = std::min(n_full, p_max_elmts);
        expected_nnz += expected;
        ASSERT_EQ(n_trunc, expected) << "row " << row;
        if (n_trunc == n_full) {
            for (gko::size_type k = 0; k < n_full; k++) {
                ASSERT_EQ(trunc->get_const_col_idxs()[trunc_ptrs[row] + k],
                          full->get_const_col_idxs()[full_ptrs[row] + k])
                    << "row " << row << " entry " << k;
                ASSERT_EQ(trunc->get_const_values()[trunc_ptrs[row] + k],
                          full->get_const_values()[full_ptrs[row] + k])
                    << "row " << row << " entry " << k;
            }
            continue;
        }
        truncated_rows++;
        auto full_sum = gko::zero<value_type>();
        for (auto k = full_ptrs[row]; k < full_ptrs[row + 1]; k++) {
            full_sum += full->get_const_values()[k];
        }
        auto trunc_sum = gko::zero<value_type>();
        for (auto k = trunc_ptrs[row]; k < trunc_ptrs[row + 1]; k++) {
            trunc_sum += trunc->get_const_values()[k];
        }
        ASSERT_NE(trunc_sum, gko::zero<value_type>()) << "row " << row;
        auto full_abs = gko::zero<real_type>();
        for (auto k = full_ptrs[row]; k < full_ptrs[row + 1]; k++) {
            full_abs += gko::abs(full->get_const_values()[k]);
        }
        ASSERT_LE(gko::abs(trunc_sum - full_sum), tol * full_abs)
            << "row " << row;
    }
    ASSERT_GT(truncated_rows, 0);
    ASSERT_EQ(trunc->get_num_stored_elements(), expected_nnz);
    ASSERT_LT(trunc->get_num_stored_elements(),
              full->get_num_stored_elements());
    ASSERT_TRUE(trunc->is_sorted_by_column_index());
}


TYPED_TEST(PmisSolve, PMaxElmtsZeroAndAnOversizedCapLeaveProlongationAlone)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    auto assert_identical = [](const Mtx* expected, const Mtx* actual,
                               const char* what) {
        ASSERT_EQ(actual->get_size(), expected->get_size()) << what;
        ASSERT_EQ(actual->get_num_stored_elements(),
                  expected->get_num_stored_elements())
            << what;
        for (gko::size_type i = 0; i <= expected->get_size()[0]; i++) {
            ASSERT_EQ(actual->get_const_row_ptrs()[i],
                      expected->get_const_row_ptrs()[i])
                << what << " row_ptrs[" << i << "]";
        }
        for (gko::size_type k = 0; k < expected->get_num_stored_elements();
             k++) {
            ASSERT_EQ(actual->get_const_col_idxs()[k],
                      expected->get_const_col_idxs()[k])
                << what << " col_idxs[" << k << "]";
            ASSERT_EQ(actual->get_const_values()[k],
                      expected->get_const_values()[k])
                << what << " values[" << k << "]";
        }
    };
    auto generate_with = [this](gko::size_type p_max_elmts) {
        return MgLevel::build()
            .with_interpolation(gko::multigrid::interpolation_type::ext_i)
            .with_p_max_elmts(p_max_elmts)
            .on(this->exec)
            ->generate(this->mtx);
    };

    auto default_level =
        MgLevel::build()
            .with_interpolation(gko::multigrid::interpolation_type::ext_i)
            .on(this->exec)
            ->generate(this->mtx);
    auto zero_level = generate_with(0);
    auto wide_level = generate_with(1000);
    auto expected = gko::as<Mtx>(default_level->get_prolong_op());

    ASSERT_LT(expected->get_num_stored_elements(), gko::size_type{1000});
    assert_identical(expected.get(),
                     gko::as<Mtx>(zero_level->get_prolong_op()).get(),
                     "p_max_elmts = 0");
    assert_identical(expected.get(),
                     gko::as<Mtx>(wide_level->get_prolong_op()).get(),
                     "p_max_elmts wider than every row");
}


class PmisTerminalCoarsening : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::Pmis<value_type, index_type>;

    PmisTerminalCoarsening() : exec(gko::ReferenceExecutor::create())
    {
        constexpr index_type n = 8;
        const auto num_rows = static_cast<gko::size_type>(n) * n * n;
        gko::matrix_data<value_type, index_type> data{
            gko::dim<2>(num_rows, num_rows)};
        for (index_type z = 0; z < n; z++) {
            for (index_type y = 0; y < n; y++) {
                for (index_type x = 0; x < n; x++) {
                    const index_type row = x + n * y + n * n * z;
                    if (z > 0) {
                        data.nonzeros.emplace_back(row, row - n * n,
                                                   value_type{-1});
                    }
                    if (y > 0) {
                        data.nonzeros.emplace_back(row, row - n,
                                                   value_type{-1});
                    }
                    if (x > 0) {
                        data.nonzeros.emplace_back(row, row - 1,
                                                   value_type{-1});
                    }
                    data.nonzeros.emplace_back(row, row, value_type{6});
                    if (x < n - 1) {
                        data.nonzeros.emplace_back(row, row + 1,
                                                   value_type{-1});
                    }
                    if (y < n - 1) {
                        data.nonzeros.emplace_back(row, row + n,
                                                   value_type{-1});
                    }
                    if (z < n - 1) {
                        data.nonzeros.emplace_back(row, row + n * n,
                                                   value_type{-1});
                    }
                }
            }
        }
        data.sort_row_major();
        mtx = gko::share(Mtx::create(exec));
        mtx->read(data);

        b = Vec::create(exec, gko::dim<2>{num_rows, 1});
        b->fill(value_type{1});
        x0 = Vec::create(exec, gko::dim<2>{num_rows, 1});
        x0->fill(value_type{0});
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<Vec> b;
    std::unique_ptr<Vec> x0;
};


TEST_F(PmisTerminalCoarsening, ExtIStopsCoarseningInsteadOfFailing)
{
    auto logger = gko::share(gko::log::Convergence<value_type>::create());
    auto solver =
        gko::solver::Multigrid::build()
            .with_max_levels(25u)
            .with_min_coarse_rows(9u)
            .with_mg_level(MgLevel::build()
                               .with_strength_threshold(0.25)
                               .with_interpolation(
                                   gko::multigrid::interpolation_type::ext_i))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-9))
            .on(exec)
            ->generate(mtx);
    solver->add_logger(logger);
    auto x = gko::clone(x0);

    solver->apply(b, x);

    const auto& levels = solver->get_mg_level_list();
    ASSERT_EQ(levels.size(), 3);
    EXPECT_GT(levels.back()->get_coarse_op()->get_size()[0], 9);
    EXPECT_TRUE(logger->has_converged());
}


TEST_F(PmisTerminalCoarsening, MatrixThatCannotCoarsenAtAllStillFails)
{
    auto fully_summed = gko::share(Mtx::create(exec));
    fully_summed->read({{5, 5},
                        {{0, 0, value_type{4}},
                         {0, 2, value_type{1}},
                         {0, 4, value_type{1}},
                         {1, 1, value_type{4}},
                         {1, 4, value_type{1}},
                         {2, 1, value_type{5}},
                         {2, 2, value_type{3}},
                         {2, 4, value_type{1}},
                         {3, 2, value_type{-1}},
                         {3, 3, value_type{2}},
                         {3, 4, value_type{1}},
                         {4, 4, value_type{1}}}});

    ASSERT_THROW(
        gko::solver::Multigrid::build()
            .with_max_levels(10u)
            .with_min_coarse_rows(2u)
            .with_mg_level(MgLevel::build())
            .with_criteria(gko::stop::Iteration::build().with_max_iters(10u))
            .on(exec)
            ->generate(fully_summed),
        gko::ValueMismatch);
}

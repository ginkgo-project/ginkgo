// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/solver_debug.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


template <typename T>
class SolverDebug : public ::testing::Test {
public:
    using Dense = gko::matrix::Dense<T>;
    using Cg = gko::solver::Cg<T>;

    SolverDebug() : ref{gko::ReferenceExecutor::create()}
    {
        mtx = gko::initialize<Dense>({T{1.0}}, ref);
        in = gko::initialize<Dense>({T{2.0}}, ref);
        out = gko::initialize<Dense>({T{4.0}}, ref);
        zero = gko::initialize<Dense>({T{0.0}}, ref);
        solver =
            Cg::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .on(ref)
                ->generate(mtx);
    }

    template <typename Mtx>
    void assert_file_equals(const std::string& filename, Mtx* ref_mtx)
    {
        auto cleanup = [filename] {
            std::remove((filename + ".mtx").c_str());
            std::remove((filename + ".bin").c_str());
        };
        if (!ref_mtx) {
            cleanup();
            return;
        }
        SCOPED_TRACE(filename);
        std::ifstream stream_mtx{filename + ".mtx"};
        std::ifstream stream_bin{filename + ".bin", std::ios::binary};
        auto mtx = gko::read<Dense>(stream_mtx, ref);
        auto mtx_bin = gko::read_binary<Dense>(stream_bin, ref);
        cleanup();
        GKO_ASSERT_MTX_NEAR(mtx, ref_mtx, 0.0);
        GKO_ASSERT_MTX_NEAR(mtx_bin, ref_mtx, 0.0);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<Dense> mtx;
    std::shared_ptr<Dense> in;
    std::unique_ptr<Dense> out;
    std::unique_ptr<Dense> zero;
    std::unique_ptr<Cg> solver;
};

TYPED_TEST_SUITE(SolverDebug, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(SolverDebug, TableWorks)
{
    using T = TypeParam;
    std::stringstream ref_ss;
    int default_column_width = 12;
    auto dynamic_type = gko::name_demangling::get_dynamic_type(*this->solver);
    ref_ss << dynamic_type << "::apply(" << this->in.get() << ','
           << this->out.get() << ") of dimensions " << this->solver->get_size()
           << " and " << this->in->get_size()[1] << " rhs\n";
    ref_ss << std::setw(default_column_width) << "Iteration"
           << std::setw(default_column_width) << "alpha"
           << std::setw(default_column_width) << "beta"
           << std::setw(default_column_width) << "prev_rho"
           << std::setw(default_column_width) << "rho"
           << std::setw(default_column_width) << "implicit_sq_residual_norm"
           << '\n';
    ref_ss << std::setw(default_column_width) << 0
           << std::setw(default_column_width) << T{0.0}
           << std::setw(default_column_width) << T{0.0}
           << std::setw(default_column_width) << T{1.0}
           << std::setw(default_column_width) << T{4.0}
           << std::setw(default_column_width) << T{4.0} << '\n'
           << std::setw(default_column_width) << 1
           << std::setw(default_column_width) << T{0.0}
           << std::setw(default_column_width) << T{4.0}
           << std::setw(default_column_width) << T{0.0}
           << std::setw(default_column_width) << T{4.0}
           << std::setw(default_column_width) << T{0.0} << '\n';
    std::stringstream ss;
    this->solver->add_logger(gko::log::SolverDebug::create_scalar_table(ss));

    this->solver->apply(this->in, this->out);

    ASSERT_EQ(ss.str(), ref_ss.str());
}


TYPED_TEST(SolverDebug, CsvWorks)
{
    using T = TypeParam;
    std::stringstream ref_ss;
    auto dynamic_type = gko::name_demangling::get_dynamic_type(*this->solver);
    ref_ss << dynamic_type << "::apply(" << this->in.get() << ','
           << this->out.get() << ") of dimensions " << this->solver->get_size()
           << " and " << this->in->get_size()[1] << " rhs\n";
    ref_ss << "Iteration,alpha,beta,prev_rho,rho,implicit_sq_residual_norm"
           << '\n';
    ref_ss << 0 << ',' << T{0.0} << ',' << T{0.0} << ',' << T{1.0} << ','
           << T{4.0} << ',' << T{4.0} << '\n'
           << 1 << ',' << T{0.0} << ',' << T{4.0} << ',' << T{0.0} << ','
           << T{4.0} << ',' << T{0.0} << '\n';
    std::stringstream ss;
    this->solver->add_logger(gko::log::SolverDebug::create_scalar_csv(ss));

    this->solver->apply(this->in, this->out);

    ASSERT_EQ(ss.str(), ref_ss.str());
}


TYPED_TEST(SolverDebug, StorageWorks)
{
    using T = TypeParam;
    using Dense = typename TestFixture::Dense;
    auto orig_out = this->out->clone();
    auto init_residual = gko::initialize<Dense>({T{-2.0}}, this->ref);
    std::vector<std::pair<std::string, Dense*>> files{
        {"solver_debug_test_0_alpha", this->zero.get()},
        {"solver_debug_test_0_beta", nullptr},
        {"solver_debug_test_0_implicit_sq_residual_norm", orig_out.get()},
        {"solver_debug_test_0_minus_one", nullptr},
        {"solver_debug_test_0_one", nullptr},
        {"solver_debug_test_0_p", nullptr},
        {"solver_debug_test_0_prev_rho", nullptr},
        {"solver_debug_test_0_q", nullptr},
        {"solver_debug_test_0_r", nullptr},
        {"solver_debug_test_0_residual", init_residual.get()},
        {"solver_debug_test_0_rho", nullptr},
        {"solver_debug_test_0_solution", orig_out.get()},
        {"solver_debug_test_0_z", nullptr},
        {"solver_debug_test_1_alpha", nullptr},
        {"solver_debug_test_1_beta", nullptr},
        {"solver_debug_test_1_implicit_sq_residual_norm", this->zero.get()},
        {"solver_debug_test_1_minus_one", nullptr},
        {"solver_debug_test_1_one", nullptr},
        {"solver_debug_test_1_p", nullptr},
        {"solver_debug_test_1_prev_rho", nullptr},
        {"solver_debug_test_1_q", nullptr},
        {"solver_debug_test_1_r", nullptr},
        {"solver_debug_test_1_residual", this->zero.get()},
        {"solver_debug_test_1_rho", nullptr},
        {"solver_debug_test_1_solution", this->in.get()},
        {"solver_debug_test_1_z", nullptr},
        {"solver_debug_test_initial_guess", orig_out.get()},
        {"solver_debug_test_rhs", this->in.get()},
        {"solver_debug_test_system_matrix", this->mtx.get()}};
    this->solver->add_logger(gko::log::SolverDebug::create_vector_storage(
        "solver_debug_test", false));
    this->solver->add_logger(gko::log::SolverDebug::create_vector_storage(
        "solver_debug_test", true));

    this->solver->apply(this->in, this->out);

    for (auto pair : files) {
        this->assert_file_equals(pair.first, pair.second);
    }
}

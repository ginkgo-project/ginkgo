// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/solver_debug.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/test/utils.hpp"


template <typename T>
class SolverDebug : public ::testing::Test {
public:
    using Dense = gko::matrix::Dense<T>;
    using Cg = gko::solver::Cg<T>;

    SolverDebug() : ref{gko::ReferenceExecutor::create()}
    {
        mtx = gko::initialize<Dense>({T{1.0}}, ref);
        in = gko::initialize<Dense>({T{2.0}}, ref);
        out = mtx->clone();
        solver =
            Cg::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .on(ref)
                ->generate(mtx);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<Dense> mtx;
    std::shared_ptr<Dense> in;
    std::unique_ptr<Dense> out;
    std::unique_ptr<Cg> solver;
};

TYPED_TEST_SUITE(SolverDebug, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(SolverDebug, Works)
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
           << std::setw(default_column_width) << "rho" << '\n';
    ref_ss << std::setw(default_column_width) << 0
           << std::setw(default_column_width) << T{0.0}
           << std::setw(default_column_width) << T{0.0}
           << std::setw(default_column_width) << T{1.0}
           << std::setw(default_column_width) << T{1.0} << '\n'
           << std::setw(default_column_width) << 1
           << std::setw(default_column_width) << T{0.0}
           << std::setw(default_column_width) << T{1.0}
           << std::setw(default_column_width) << T{0.0}
           << std::setw(default_column_width) << T{1.0} << '\n';
    std::stringstream ss;
    this->solver->add_logger(gko::log::SolverDebug::create(ss));

    this->solver->apply(this->in, this->out);

    ASSERT_EQ(ss.str(), ref_ss.str());
}

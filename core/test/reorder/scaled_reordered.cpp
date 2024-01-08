// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/scaled_reordered.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/reorder/rcm.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


namespace {


class ScaledReorderedFactory : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using diag = gko::matrix::Diagonal<value_type>;
    using solver_type = gko::solver::Bicgstab<value_type>;
    using reorder_type = gko::reorder::Rcm<value_type, index_type>;
    using scaled_reordered_type =
        gko::experimental::reorder::ScaledReordered<value_type, index_type>;

    ScaledReorderedFactory()
        : exec(gko::ReferenceExecutor::create()),
          scaled_reordered_factory(scaled_reordered_type::build().on(exec)),
          reordering_factory(reorder_type::build().on(exec)),
          solver_factory(solver_type::build().on(exec)),
          diag_matrix(diag::create(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<diag> diag_matrix;
    std::shared_ptr<typename scaled_reordered_type::Factory>
        scaled_reordered_factory;
    std::shared_ptr<typename reorder_type::Factory> reordering_factory;
    std::shared_ptr<typename solver_type::Factory> solver_factory;
};


TEST_F(ScaledReorderedFactory, KnowsItsExecutor)
{
    auto scaled_reordered_factory =
        scaled_reordered_type::build().on(this->exec);

    ASSERT_EQ(scaled_reordered_factory->get_executor(), this->exec);
}


TEST_F(ScaledReorderedFactory, CanSetReorderingFactory)
{
    auto scaled_reordered_factory =
        scaled_reordered_type::build()
            .with_reordering(this->reordering_factory)
            .on(this->exec);

    ASSERT_EQ(scaled_reordered_factory->get_parameters().reordering,
              this->reordering_factory);
}


TEST_F(ScaledReorderedFactory, CanSetInnerOperatorFactory)
{
    auto scaled_reordered_factory =
        scaled_reordered_type::build()
            .with_inner_operator(this->solver_factory)
            .on(this->exec);

    ASSERT_EQ(scaled_reordered_factory->get_parameters().inner_operator,
              this->solver_factory);
}


TEST_F(ScaledReorderedFactory, CanSetRowScaling)
{
    auto scaled_reordered_factory = scaled_reordered_type::build()
                                        .with_row_scaling(this->diag_matrix)
                                        .on(this->exec);

    ASSERT_EQ(scaled_reordered_factory->get_parameters().row_scaling,
              this->diag_matrix);
}


TEST_F(ScaledReorderedFactory, CanSetColScaling)
{
    auto scaled_reordered_factory = scaled_reordered_type::build()
                                        .with_col_scaling(this->diag_matrix)
                                        .on(this->exec);

    ASSERT_EQ(scaled_reordered_factory->get_parameters().col_scaling,
              this->diag_matrix);
}


}  // namespace


GKO_END_DISABLE_DEPRECATION_WARNINGS

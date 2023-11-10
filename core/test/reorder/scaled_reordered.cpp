/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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

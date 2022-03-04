/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/multigrid/selection.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SelectionFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::Selection<value_type, index_type>;
    SelectionFactory()
        : exec(gko::ReferenceExecutor::create()),
          selection1_factory(
              MgLevel::build().with_num_jumps(4u).with_skip_sorting(true).on(
                  exec)),
          selection2_factory(
              MgLevel::build()
                  .with_coarse_rows(gko::Array<index_type>(exec, {2, 3}))
                  .with_skip_sorting(true)
                  .on(exec))

    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename MgLevel::Factory> selection1_factory;
    std::unique_ptr<typename MgLevel::Factory> selection2_factory;
};

TYPED_TEST_SUITE(SelectionFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(SelectionFactory, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->selection1_factory->get_executor(), this->exec);
}


TYPED_TEST(SelectionFactory, DefaultSetting)
{
    using MgLevel = typename TestFixture::MgLevel;
    auto factory = MgLevel::build().on(this->exec);

    ASSERT_EQ(factory->get_parameters().num_jumps, 2u);
    ASSERT_EQ(factory->get_parameters().coarse_rows.get_data(), nullptr);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
}


TYPED_TEST(SelectionFactory, SetCoarseRows)
{
    using T = typename TestFixture::index_type;
    GKO_ASSERT_ARRAY_EQ(this->selection2_factory->get_parameters().coarse_rows,
                        gko::Array<T>(this->exec, {2, 3}));
}


TYPED_TEST(SelectionFactory, SetNumJumps)
{
    ASSERT_EQ(this->selection1_factory->get_parameters().num_jumps, 4u);
}


TYPED_TEST(SelectionFactory, SetSkipSorting)
{
    ASSERT_EQ(this->selection1_factory->get_parameters().skip_sorting, true);
}


}  // namespace

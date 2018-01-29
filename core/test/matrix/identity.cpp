/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/matrix/identity.hpp>


#include <gtest/gtest.h>


#include <core/matrix/dense.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Identity : public ::testing::Test {
protected:
    using Id = gko::matrix::Identity<>;
    using Vec = gko::matrix::Dense<>;

    Identity() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};


TEST_F(Identity, CanBeEmpty)
{
    auto empty = Id::create(exec);
    ASSERT_EQ(empty->get_num_rows(), 0);
    ASSERT_EQ(empty->get_num_cols(), 0);
    ASSERT_EQ(empty->get_num_stored_elements(), 0);
}


TEST_F(Identity, CanBeConstructedWithSize)
{
    auto identity = Id::create(exec, 5);
    ASSERT_EQ(identity->get_num_rows(), 5);
    ASSERT_EQ(identity->get_num_cols(), 5);
    ASSERT_EQ(identity->get_num_stored_elements(), 0);
}


TEST_F(Identity, AppliesToVector)
{
    auto identity = Id::create(exec, 3);
    auto x = Vec::create(exec, 3, 1);
    auto b = gko::initialize<Vec>({2.0, 1.0, 5.0}, exec);

    identity->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({2.0, 1.0, 5.0}), 0.0);
}


TEST_F(Identity, AppliesToMultipleVectors)
{
    auto identity = Id::create(exec, 3);
    auto x = Vec::create(exec, 3, 2, 3);
    auto b =
        gko::initialize<Vec>(3, {{2.0, 3.0}, {1.0, 2.0}, {5.0, -1.0}}, exec);

    identity->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{2.0, 3.0}, {1.0, 2.0}, {5.0, -1.0}}), 0.0);
}


TEST(IdentityFactory, CanGenerateIdentityMatrix)
{
    auto exec = gko::ReferenceExecutor::create();
    auto id_factory = gko::matrix::IdentityFactory<>::create(exec);
    auto mtx = gko::matrix::Dense<>::create(exec, 5, 5);

    auto id = id_factory->generate(std::move(mtx));

    ASSERT_EQ(id->get_num_rows(), 5);
    ASSERT_EQ(id->get_num_cols(), 5);
    ASSERT_EQ(id->get_num_stored_elements(), 0);
}


}  // namespace

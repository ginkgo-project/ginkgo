/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
                                                               modification, are
permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
                documentation and/or other materials provided with the
distribution.

           3. Neither the name of the copyright holder nor the names of its
               contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
        IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
        TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
        PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
            HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
                                                                LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
            ******************************<GINKGO
LICENSE>*******************************/

#include "core/constraints/constraints_handler_kernels.hpp"


#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


template <typename ValueIndexType>
class ZeroRowsStrategy : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using dense = gko::matrix::Dense<value_type>;
    using mtx = gko::matrix::Csr<value_type, index_type>;
    // using strategy = gko::constraints::ZeroRowsStrategy<value_type,
    // index_type>;

    ZeroRowsStrategy()
        : ref(gko::ReferenceExecutor::create()),
          strategy(),
          empty_idxs(ref),
          empty_mtx(gko::share(mtx::create(ref))),
          empty_values(gko::share(dense::create(ref))),
          empty_rhs(gko::share(dense::create(ref))),
          empty_init(gko::share(dense::create(ref))),
          def_idxs(ref, {0, 2}),
          def_csr(gko::share(gko::initialize<mtx>(
              {{1, 0, 2, 3}, {0, 4, 0, 0}, {0, 5, 6, 0}, {7, 0, 0, 8}},
              this->ref)))
    {}


    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::constraints::ZeroRowsStrategy<value_type, index_type> strategy;

    gko::Array<index_type> empty_idxs;
    std::shared_ptr<mtx> empty_mtx;
    std::shared_ptr<dense> empty_values;
    std::shared_ptr<dense> empty_rhs;
    std::shared_ptr<dense> empty_init;

    gko::Array<index_type> def_idxs;
    std::shared_ptr<mtx> def_csr;
};

TYPED_TEST_SUITE(ZeroRowsStrategy, gko::test::ValueIndexTypes);


TYPED_TEST(ZeroRowsStrategy, ConstructOperatorFromEmptyIndicesAndMatrix)
{
    auto result = this->strategy.construct_operator(this->empty_idxs,
                                                    this->empty_mtx.get());

    ASSERT_EQ(result.get(), this->empty_mtx.get());
}

TYPED_TEST(ZeroRowsStrategy, ConstructOperator)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx = typename TestFixture::mtx;
    auto csr = gko::initialize<mtx>(
        {{1, 0, 2, 3}, {0, 4, 0, 0}, {0, 5, 6, 0}, {7, 0, 0, 8}}, this->ref);
    auto result = gko::initialize<mtx>(
        {{1, 0, 0, 0}, {0, 4, 0, 0}, {0, 0, 1, 0}, {7, 0, 0, 8}}, this->ref);
    gko::Array<index_type> subset{this->ref, {0, 2}};

    auto cons = this->strategy.construct_operator(subset, csr.get());

    GKO_ASSERT_MTX_NEAR(gko::as<mtx>(cons.get()), result.get(), 0);
}


TYPED_TEST(ZeroRowsStrategy, ConstructRhsFromEmpty)
{
    auto rhs = this->strategy.construct_right_hand_side(
        this->empty_idxs, this->empty_mtx.get(), this->empty_init.get(),
        this->empty_rhs.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(rhs->get_size(), gko::dim<2>(0, 0));
}


TYPED_TEST(ZeroRowsStrategy, ConstructRhs)
{
    using dense = typename TestFixture::dense;
    auto orig_rhs = gko::initialize<dense>({1, 0, 1, 0}, this->ref);
    auto orig_init = gko::initialize<dense>({1, 2, 1, 2}, this->ref);
    auto result = gko::initialize<dense>({0, -8, 0, -23}, this->ref);

    auto rhs = this->strategy.construct_right_hand_side(
        this->def_idxs, this->def_csr.get(), orig_init.get(), orig_rhs.get());

    GKO_ASSERT_MTX_NEAR(result.get(), gko::as<dense>(rhs.get()), 0);
}

TYPED_TEST(ZeroRowsStrategy, ConstructInitFromEmpty)
{
    auto init = this->strategy.construct_initial_guess(
        this->empty_idxs, this->empty_mtx.get(), this->empty_init.get(),
        this->empty_values.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(init->get_size(), gko::dim<2>(0, 0));
}


TYPED_TEST(ZeroRowsStrategy, ConstructInit)
{
    using dense = typename TestFixture::dense;
    auto values = gko::initialize<dense>({1, -11, 1, -11}, this->ref);
    auto orig_init = gko::initialize<dense>({1, 2, 1, 2}, this->ref);
    auto result = gko::initialize<dense>({0, 2, 0, 2}, this->ref);

    auto init = this->strategy.construct_initial_guess(
        this->def_idxs, this->def_csr.get(), orig_init.get(), values.get());

    GKO_ASSERT_MTX_NEAR(result.get(), gko::as<dense>(init.get()), 0);
}


TYPED_TEST(ZeroRowsStrategy, UpdateSolution)
{
    using dense = typename TestFixture::dense;
    auto values = gko::initialize<dense>({1, -11, 1, -11}, this->ref);
    auto orig_init = gko::initialize<dense>({1, 2, 1, 2}, this->ref);
    auto solution = gko::initialize<dense>({0, 55, 0, 55}, this->ref);
    auto result = gko::initialize<dense>({1, 57, 1, 57}, this->ref);

    this->strategy.correct_solution(this->def_idxs, values.get(),
                                    orig_init.get(), solution.get());

    GKO_ASSERT_MTX_NEAR(result.get(), solution.get(), 0);
}


TYPED_TEST(ZeroRowsStrategy, UpdateSolutionFromInitWithoutValues)
{
    using dense = typename TestFixture::dense;
    auto values = gko::initialize<dense>({1, -11, 1, -11}, this->ref);
    auto orig_init = gko::initialize<dense>({-3, 2, -3, 2}, this->ref);
    auto solution = gko::initialize<dense>({0, 55, 0, 55}, this->ref);
    auto result = gko::initialize<dense>({1, 57, 1, 57}, this->ref);

    this->strategy.correct_solution(this->def_idxs, values.get(),
                                    orig_init.get(), solution.get());

    GKO_ASSERT_MTX_NEAR(result.get(), solution.get(), 0);
}


template <typename ValueIndexType>
class ConstrainedSystem : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using dense = gko::matrix::Dense<value_type>;
    using mtx = gko::matrix::Csr<value_type, index_type>;
    using handler =
        gko::constraints::ConstraintsHandler<value_type, index_type>;

    ConstrainedSystem() : ref(gko::ReferenceExecutor::create()) {}


    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(ConstrainedSystem, gko::test::ValueIndexTypes);

TYPED_TEST(ConstrainedSystem, CanCreateWithIdxsMatrix)
{
    using index_type = typename TestFixture::index_type;
    using mtx = typename TestFixture::mtx;
    using handler = typename TestFixture::handler;
    gko::Array<index_type> idxs{this->ref};
    auto csr = gko::share(mtx::create(this->ref));

    handler ch(idxs, csr);

    GKO_ASSERT_ARRAY_EQ(*ch.get_constrained_indices(), idxs);
    ASSERT_EQ(ch.get_orig_operator(), csr.get());
}


}  // namespace

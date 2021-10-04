/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/constraints/constraints_handler_kernels.hpp"
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
          def_mtx(gko::share(gko::initialize<mtx>(
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
    std::shared_ptr<mtx> def_mtx;
};

TYPED_TEST_SUITE(ZeroRowsStrategy, gko::test::ValueIndexTypes);


TYPED_TEST(ZeroRowsStrategy, DoesNotOwnOriginalOperator)
{
    {
        this->strategy.construct_operator(this->empty_idxs, this->empty_mtx);
    }

    ASSERT_NO_FATAL_FAILURE(this->empty_mtx->get_size());
}


TYPED_TEST(ZeroRowsStrategy, ConstructOperatorFromEmptyIndicesAndMatrix)
{
    auto result =
        this->strategy.construct_operator(this->empty_idxs, this->empty_mtx);

    ASSERT_EQ(result.get(), this->empty_mtx.get());
}

TYPED_TEST(ZeroRowsStrategy, ConstructOperator)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx = typename TestFixture::mtx;
    auto csr = gko::share(gko::initialize<mtx>(
        {{1, 0, 2, 3}, {0, 4, 0, 0}, {0, 5, 6, 0}, {7, 0, 0, 8}}, this->ref));
    auto result = gko::initialize<mtx>(
        {{1, 0, 0, 0}, {0, 4, 0, 0}, {0, 0, 1, 0}, {7, 0, 0, 8}}, this->ref);
    gko::Array<index_type> subset{this->ref, {0, 2}};

    auto cons = this->strategy.construct_operator(subset, csr);

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
        this->def_idxs, this->def_mtx.get(), orig_init.get(), orig_rhs.get());

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
        this->def_idxs, this->def_mtx.get(), orig_init.get(), values.get());

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

struct apply_counter {
    int op = 0;
    int rhs = 0;
    int init = 0;
    int sol = 0;
};


template <typename ValueType, typename IndexType>
class StrategyWithCounter
    : public gko::constraints::ZeroRowsStrategy<ValueType, IndexType> {
public:
    StrategyWithCounter(std::shared_ptr<apply_counter> c_) : c(std::move(c_)) {}

    std::shared_ptr<gko::LinOp> construct_operator(
        const gko::Array<IndexType>& idxs,
        std::shared_ptr<gko::LinOp> op) override
    {
        c->op++;
        return gko::constraints::ZeroRowsStrategy<
            ValueType, IndexType>::construct_operator(idxs, op);
    }
    std::unique_ptr<gko::LinOp> construct_right_hand_side(
        const gko::Array<IndexType>& idxs, const gko::LinOp* op,
        const gko::matrix::Dense<ValueType>* init_guess,
        const gko::matrix::Dense<ValueType>* rhs) override
    {
        c->rhs++;
        return gko::constraints::ZeroRowsStrategy<
            ValueType, IndexType>::construct_right_hand_side(idxs, op,
                                                             init_guess, rhs);
    }
    std::unique_ptr<gko::LinOp> construct_initial_guess(
        const gko::Array<IndexType>& idxs, const gko::LinOp* op,
        const gko::matrix::Dense<ValueType>* init_guess,
        const gko::matrix::Dense<ValueType>* constrained_values) override
    {
        c->init++;
        return gko::constraints::ZeroRowsStrategy<
            ValueType, IndexType>::construct_initial_guess(idxs, op, init_guess,
                                                           constrained_values);
    }
    void correct_solution(
        const gko::Array<IndexType>& idxs,
        const gko::matrix::Dense<ValueType>* constrained_values,
        const gko::matrix::Dense<ValueType>* orig_init_guess,
        gko::matrix::Dense<ValueType>* solution) override
    {
        c->sol++;
        gko::constraints::ZeroRowsStrategy<
            ValueType, IndexType>::correct_solution(idxs, constrained_values,
                                                    orig_init_guess, solution);
    }

    std::shared_ptr<apply_counter> c;
};

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


    ConstrainedSystem()
        : ref(gko::ReferenceExecutor::create()),
          strategy(),
          empty_idxs(ref),
          empty_mtx(gko::share(mtx::create(ref))),
          empty_values(gko::share(dense::create(ref))),
          empty_rhs(gko::share(dense::create(ref))),
          empty_init(gko::share(dense::create(ref))),
          def_idxs(ref, {0, 2}),
          def_mtx(gko::share(gko::initialize<mtx>(
              {{1, 0, 2, 3}, {0, 4, 0, 0}, {0, 5, 6, 0}, {7, 0, 0, 8}},
              this->ref))),
          empty_handler(this->empty_idxs, this->empty_mtx),
          counter(std::make_shared<apply_counter>()),
          counted_handler(
              this->empty_idxs, this->empty_mtx, this->empty_values,
              this->empty_rhs, this->empty_init,
              std::make_unique<StrategyWithCounter<value_type, index_type>>(
                  counter))
    {}


    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::constraints::ZeroRowsStrategy<value_type, index_type> strategy;

    gko::Array<index_type> empty_idxs;
    std::shared_ptr<mtx> empty_mtx;
    std::shared_ptr<dense> empty_values;
    std::shared_ptr<dense> empty_rhs;
    std::shared_ptr<dense> empty_init;

    gko::Array<index_type> def_idxs;
    std::shared_ptr<mtx> def_mtx;

    handler empty_handler;

    std::shared_ptr<apply_counter> counter;
    handler counted_handler;
};


TYPED_TEST_SUITE(ConstrainedSystem, gko::test::ValueIndexTypes);

TYPED_TEST(ConstrainedSystem, CanCreateWithIdxsMatrix)
{
    using handler = typename TestFixture::handler;
    handler ch(this->empty_idxs, this->empty_mtx);

    GKO_ASSERT_ARRAY_EQ(*ch.get_constrained_indices(), this->empty_idxs);
    ASSERT_EQ(ch.get_orig_operator(), this->empty_mtx.get());
}

TYPED_TEST(ConstrainedSystem, CanCreateWithFullSystem)
{
    using handler = typename TestFixture::handler;

    handler ch(this->empty_idxs, this->empty_mtx, this->empty_values,
               this->empty_rhs, this->empty_init);

    GKO_ASSERT_ARRAY_EQ(*ch.get_constrained_indices(), this->empty_idxs);
    ASSERT_EQ(ch.get_orig_operator(), this->empty_mtx.get());
    ASSERT_EQ(ch.get_constrained_values(), this->empty_values.get());
    ASSERT_EQ(ch.get_orig_right_hand_side(), this->empty_rhs.get());
    ASSERT_EQ(ch.get_orig_initial_guess(), this->empty_init.get());
}

TYPED_TEST(ConstrainedSystem, CanCreateWithoutInitialGuess)
{
    using handler = typename TestFixture::handler;

    handler ch(this->empty_idxs, this->empty_mtx, this->empty_values,
               this->empty_rhs);

    GKO_ASSERT_ARRAY_EQ(*ch.get_constrained_indices(), this->empty_idxs);
    ASSERT_EQ(ch.get_orig_operator(), this->empty_mtx.get());
    ASSERT_EQ(ch.get_constrained_values(), this->empty_values.get());
    ASSERT_EQ(ch.get_orig_right_hand_side(), this->empty_rhs.get());
    ASSERT_FALSE(ch.get_orig_initial_guess());
}

TYPED_TEST(ConstrainedSystem, UpdateEmptyHandlerWithValues)
{
    this->empty_handler.with_constrained_values(this->empty_values);

    ASSERT_EQ(this->empty_handler.get_constrained_values(),
              this->empty_values.get());
    ASSERT_FALSE(this->empty_handler.get_orig_right_hand_side());
    ASSERT_FALSE(this->empty_handler.get_orig_initial_guess());
}

TYPED_TEST(ConstrainedSystem, UpdateEmptyHandlerWithRhs)
{
    this->empty_handler.with_right_hand_side(this->empty_rhs);

    ASSERT_FALSE(this->empty_handler.get_constrained_values());
    ASSERT_EQ(this->empty_handler.get_orig_right_hand_side(),
              this->empty_rhs.get());
    ASSERT_FALSE(this->empty_handler.get_orig_initial_guess());
}


TYPED_TEST(ConstrainedSystem, UpdateEmptyHandlerWithInit)
{
    this->empty_handler.with_initial_guess(this->empty_init);

    ASSERT_FALSE(this->empty_handler.get_constrained_values());
    ASSERT_FALSE(this->empty_handler.get_orig_right_hand_side());
    ASSERT_EQ(this->empty_handler.get_orig_initial_guess(),
              this->empty_init.get());
}

TYPED_TEST(ConstrainedSystem, UpdateEmptyHandlerChain)
{
    this->empty_handler.with_constrained_values(this->empty_values)
        .with_initial_guess(this->empty_init)
        .with_right_hand_side(this->empty_rhs);


    ASSERT_EQ(this->empty_handler.get_constrained_values(),
              this->empty_values.get());
    ASSERT_EQ(this->empty_handler.get_orig_right_hand_side(),
              this->empty_rhs.get());
    ASSERT_EQ(this->empty_handler.get_orig_initial_guess(),
              this->empty_init.get());
}

TYPED_TEST(ConstrainedSystem, ReconstructsLazily)
{
    auto prev_counter = *this->counter;

    this->counted_handler.get_initial_guess();
    this->counted_handler.get_right_hand_side();
    this->counted_handler.get_operator();

    ASSERT_EQ(this->counter->rhs, prev_counter.rhs);
    ASSERT_EQ(this->counter->init, prev_counter.init);
    ASSERT_EQ(this->counter->op, prev_counter.op);
}


TYPED_TEST(ConstrainedSystem, CanReconstructConstrainedSystem)
{
    auto prev_counter = *this->counter;

    this->counted_handler.reconstruct_system();

    ASSERT_EQ(this->counter->rhs - prev_counter.rhs, 1);
    ASSERT_EQ(this->counter->init - prev_counter.init, 1);
    ASSERT_EQ(this->counter->op - prev_counter.op, 0);
}


TYPED_TEST(ConstrainedSystem, ReconstructsRhsAndInitAfterWithValues)
{
    auto prev_counter = *this->counter;
    auto cloned_values = gko::share(gko::clone(this->empty_values));

    this->counted_handler.with_constrained_values(cloned_values);
    this->counted_handler.get_initial_guess();
    this->counted_handler.get_right_hand_side();
    this->counted_handler.get_operator();

    ASSERT_EQ(this->counter->rhs - prev_counter.rhs, 1);
    ASSERT_EQ(this->counter->init - prev_counter.init, 1);
    ASSERT_EQ(this->counter->op - prev_counter.op, 0);
}


TYPED_TEST(ConstrainedSystem, ReconstructsOnlyRhsAfterWithRhs)
{
    auto prev_counter = *this->counter;
    auto cloned_rhs = gko::share(gko::clone(this->empty_rhs));

    this->counted_handler.with_right_hand_side(cloned_rhs);
    this->counted_handler.get_initial_guess();
    this->counted_handler.get_right_hand_side();
    this->counted_handler.get_operator();

    ASSERT_EQ(this->counter->rhs - prev_counter.rhs, 1);
    ASSERT_EQ(this->counter->init - prev_counter.init, 0);
    ASSERT_EQ(this->counter->op - prev_counter.op, 0);
}


TYPED_TEST(ConstrainedSystem, ReconstructsRhsAndInitAfterWithInit)
{
    auto prev_counter = *this->counter;
    auto cloned_init = gko::share(gko::clone(this->empty_init));

    this->counted_handler.with_initial_guess(cloned_init);
    this->counted_handler.get_initial_guess();
    this->counted_handler.get_right_hand_side();
    this->counted_handler.get_operator();

    ASSERT_EQ(this->counter->rhs - prev_counter.rhs, 1);
    ASSERT_EQ(this->counter->init - prev_counter.init, 1);
    ASSERT_EQ(this->counter->op - prev_counter.op, 0);
}


TYPED_TEST(ConstrainedSystem, ReconstructsRhsAndInitForGetRhs)
{
    auto prev_counter = *this->counter;
    auto cloned_init = gko::share(gko::clone(this->empty_init));
    auto cloned_rhs = gko::share(gko::clone(this->empty_rhs));

    this->counted_handler.with_right_hand_side(cloned_rhs);
    this->counted_handler.with_initial_guess(cloned_init);
    this->counted_handler.get_right_hand_side();

    ASSERT_EQ(this->counter->rhs - prev_counter.rhs, 1);
    ASSERT_EQ(this->counter->init - prev_counter.init, 1);
}


TYPED_TEST(ConstrainedSystem, ReconstructsOnlyRhsForGetRhs)
{
    auto prev_counter = *this->counter;
    auto cloned_init = gko::share(gko::clone(this->empty_init));
    auto cloned_rhs = gko::share(gko::clone(this->empty_rhs));

    this->counted_handler.with_right_hand_side(cloned_rhs);
    this->counted_handler.get_right_hand_side();

    ASSERT_EQ(this->counter->rhs - prev_counter.rhs, 1);
    ASSERT_EQ(this->counter->init - prev_counter.init, 0);
}


TYPED_TEST(ConstrainedSystem, ReconstructsOnlyInitForGetInit)
{
    auto prev_counter = *this->counter;
    auto cloned_init = gko::share(gko::clone(this->empty_init));
    auto cloned_rhs = gko::share(gko::clone(this->empty_rhs));

    this->counted_handler.with_right_hand_side(cloned_rhs);
    this->counted_handler.with_initial_guess(cloned_init);
    this->counted_handler.get_initial_guess();

    ASSERT_EQ(this->counter->rhs - prev_counter.rhs, 0);
    ASSERT_EQ(this->counter->init - prev_counter.init, 1);
}


}  // namespace

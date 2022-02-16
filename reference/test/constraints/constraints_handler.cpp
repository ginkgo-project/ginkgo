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
#include <ginkgo/core/constraints/constraints_handler.hpp>
#include <ginkgo/core/constraints/zero_rows.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {
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
        const gko::IndexSet<IndexType>& idxs,
        std::shared_ptr<gko::LinOp> op) override
    {
        c->op++;
        return gko::constraints::ZeroRowsStrategy<
            ValueType, IndexType>::construct_operator(idxs, op);
    }
    std::unique_ptr<gko::LinOp> construct_right_hand_side(
        const gko::IndexSet<IndexType>& idxs, const gko::LinOp* op,
        const gko::matrix::Dense<ValueType>* init_guess,
        const gko::matrix::Dense<ValueType>* rhs) override
    {
        c->rhs++;
        return gko::constraints::ZeroRowsStrategy<
            ValueType, IndexType>::construct_right_hand_side(idxs, op,
                                                             init_guess, rhs);
    }
    std::unique_ptr<gko::LinOp> construct_initial_guess(
        const gko::IndexSet<IndexType>& idxs, const gko::LinOp* op,
        const gko::matrix::Dense<ValueType>* init_guess,
        const gko::matrix::Dense<ValueType>* constrained_values) override
    {
        c->init++;
        return gko::constraints::ZeroRowsStrategy<
            ValueType, IndexType>::construct_initial_guess(idxs, op, init_guess,
                                                           constrained_values);
    }
    void correct_solution(
        const gko::IndexSet<IndexType>& idxs,
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
          def_idxs(ref, 4, {ref, {0, 2}}),
          def_mtx(gko::share(gko::initialize<mtx>(
              {{1, 0, 2, 3}, {0, 4, 0, 0}, {0, 5, 6, 0}, {7, 0, 0, 8}},
              this->ref))),
          empty_handler(this->empty_idxs, this->empty_mtx),
          counter(std::make_shared<apply_counter>()),
          counted_handler(
              this->empty_idxs, this->empty_mtx, this->empty_values,
              this->empty_rhs, this->empty_init,
              std::make_unique<StrategyWithCounter<value_type, index_type>>(
                  counter)),
          system_idxs(ref, 4, {ref, {0, 3}}),
          system_mtx(gko::share(gko::initialize<mtx>(
              {{2, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 2}},
              ref))),
          system_values(
              gko::share(gko::initialize<dense>({1, -11, -11, 1}, ref))),
          system_rhs(gko::share(
              gko::initialize<dense>({1. / 9, 1. / 9, 1. / 9, 1. / 9}, ref))),
          system_init(gko::share(gko::initialize<dense>({1, 4, 5, 1}, ref))),
          system_solution(gko::share(
              gko::initialize<dense>({1, 1 + 1. / 9, 1 + 1. / 9, 1}, ref)))
    {}


    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::constraints::ZeroRowsStrategy<value_type, index_type> strategy;

    gko::IndexSet<index_type> empty_idxs;
    std::shared_ptr<mtx> empty_mtx;
    std::shared_ptr<dense> empty_values;
    std::shared_ptr<dense> empty_rhs;
    std::shared_ptr<dense> empty_init;

    gko::IndexSet<index_type> def_idxs;
    std::shared_ptr<mtx> def_mtx;

    gko::IndexSet<index_type> system_idxs;
    std::shared_ptr<mtx> system_mtx;
    std::shared_ptr<dense> system_values;
    std::shared_ptr<dense> system_rhs;
    std::shared_ptr<dense> system_init;
    std::shared_ptr<dense> system_solution;

    handler empty_handler;

    std::shared_ptr<apply_counter> counter;
    handler counted_handler;
};


TYPED_TEST_SUITE(ConstrainedSystem, gko::test::ValueIndexTypes);

template <typename IndexType>
void assert_index_set_eq(const gko::IndexSet<IndexType>& a,
                         const gko::IndexSet<IndexType>& b)
{
    auto exec = a.get_executor();
    auto num_subsets = a.get_num_subsets();

    ASSERT_EQ(num_subsets, b.get_num_subsets());
    ASSERT_EQ(a.get_num_elems(), b.get_num_elems());
    GKO_ASSERT_ARRAY_EQ(
        gko::Array<IndexType>::view(
            exec, num_subsets, const_cast<IndexType*>(a.get_subsets_begin())),
        gko::Array<IndexType>::view(
            exec, num_subsets, const_cast<IndexType*>(b.get_subsets_begin())));
    GKO_ASSERT_ARRAY_EQ(
        gko::Array<IndexType>::view(
            exec, num_subsets, const_cast<IndexType*>(a.get_subsets_end())),
        gko::Array<IndexType>::view(
            exec, num_subsets, const_cast<IndexType*>(b.get_subsets_end())));
    GKO_ASSERT_ARRAY_EQ(gko::Array<IndexType>::view(
                            exec, num_subsets,
                            const_cast<IndexType*>(a.get_superset_indices())),
                        gko::Array<IndexType>::view(
                            exec, num_subsets,
                            const_cast<IndexType*>(b.get_superset_indices())));
}


TYPED_TEST(ConstrainedSystem, CanCreateWithIdxsMatrix)
{
    using handler = typename TestFixture::handler;
    handler ch(this->empty_idxs, this->empty_mtx);

    assert_index_set_eq(*ch.get_constrained_indices(), this->empty_idxs);
    ASSERT_EQ(ch.get_orig_operator(), this->empty_mtx.get());
}

TYPED_TEST(ConstrainedSystem, CanCreateWithFullSystem)
{
    using handler = typename TestFixture::handler;

    handler ch(this->empty_idxs, this->empty_mtx, this->empty_values,
               this->empty_rhs, this->empty_init);

    assert_index_set_eq(*ch.get_constrained_indices(), this->empty_idxs);
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

    assert_index_set_eq(*ch.get_constrained_indices(), this->empty_idxs);
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


TYPED_TEST(ConstrainedSystem, ReconstructsRhsAndInitForGetInit)
{
    auto prev_counter = *this->counter;
    auto cloned_init = gko::share(gko::clone(this->empty_init));
    auto cloned_rhs = gko::share(gko::clone(this->empty_rhs));

    this->counted_handler.with_initial_guess(cloned_init);
    this->counted_handler.get_initial_guess();

    ASSERT_EQ(this->counter->rhs - prev_counter.rhs, 1);
    ASSERT_EQ(this->counter->init - prev_counter.init, 1);
}


TYPED_TEST(ConstrainedSystem, ThrowsIfNoValues)
{
    ASSERT_THROW(this->empty_handler.get_initial_guess(), gko::InvalidState);
    ASSERT_THROW(this->empty_handler.get_right_hand_side(), gko::InvalidState);
}


TYPED_TEST(ConstrainedSystem, ThrowsIfNoRhs)
{
    this->empty_handler.with_constrained_values(this->empty_values);

    ASSERT_THROW(this->empty_handler.get_initial_guess(), gko::InvalidState);
    ASSERT_THROW(this->empty_handler.get_right_hand_side(), gko::InvalidState);
}


TYPED_TEST(ConstrainedSystem, SolveConstrainedSystemWithoutInit)
{
    using value_type = typename TestFixture::value_type;
    using dense = typename TestFixture::dense;
    using handler = typename TestFixture::handler;
    using cg = gko::solver::Cg<value_type>;
    handler system_handler(this->system_idxs, this->system_mtx,
                           this->system_values, this->system_rhs);
    auto u = gko::as<dense>(system_handler.get_initial_guess());

    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(4u).on(this->ref))
        .on(this->ref)
        ->generate(system_handler.get_operator())
        ->apply(system_handler.get_right_hand_side(), u);
    system_handler.correct_solution(u);

    GKO_ASSERT_MTX_NEAR(u, this->system_solution.get(), r<value_type>::value);
}


TYPED_TEST(ConstrainedSystem, SolveConstrainedSystemWithInit)
{
    using value_type = typename TestFixture::value_type;
    using dense = typename TestFixture::dense;
    using handler = typename TestFixture::handler;
    using cg = gko::solver::Cg<value_type>;
    handler system_handler(this->system_idxs, this->system_mtx,
                           this->system_values, this->system_rhs,
                           this->system_init);
    auto u = gko::as<dense>(system_handler.get_initial_guess());

    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(4u).on(this->ref))
        .on(this->ref)
        ->generate(system_handler.get_operator())
        ->apply(system_handler.get_right_hand_side(), u);
    system_handler.correct_solution(u);

    GKO_ASSERT_MTX_NEAR(u, this->system_solution.get(), r<value_type>::value);
}


TYPED_TEST(ConstrainedSystem, SolveConstrainedSystemLazy)
{
    using value_type = typename TestFixture::value_type;
    using dense = typename TestFixture::dense;
    using handler = typename TestFixture::handler;
    using cg = gko::solver::Cg<value_type>;
    handler system_handler(this->system_idxs, this->system_mtx);
    system_handler.with_right_hand_side(this->system_rhs)
        .with_constrained_values(this->system_values);
    auto u = gko::as<dense>(system_handler.get_initial_guess());

    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(4u).on(this->ref))
        .on(this->ref)
        ->generate(system_handler.get_operator())
        ->apply(system_handler.get_right_hand_side(), u);
    system_handler.correct_solution(u);

    GKO_ASSERT_MTX_NEAR(u, this->system_solution.get(), r<value_type>::value);
}


}  // namespace

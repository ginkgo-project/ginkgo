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

#include <ginkgo/core/base/transposition.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>


#include "core/test/utils.hpp"


namespace {


int global_step = 0;


struct DummyOperator : public gko::EnableLinOp<DummyOperator>,
                       public gko::Transposable {
    DummyOperator(std::shared_ptr<const gko::Executor> exec,
                  gko::dim<2> size = {})
        : gko::EnableLinOp<DummyOperator>(exec, size),
          exec_(exec),
          size_(size),
          step_(global_step)
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override {}

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}

    int get_step() const noexcept { return step_; }

    std::unique_ptr<LinOp> transpose() const override
    {
        return std::make_unique<DummyOperator>(exec_, gko::transpose(size_));
    }

    std::unique_ptr<LinOp> conj_transpose() const override
    {
        return std::make_unique<DummyOperator>(exec_, gko::transpose(size_));
    }

private:
    std::shared_ptr<const gko::Executor> exec_;
    gko::dim<2> size_;
    int step_;
};


struct NoTransposableOperator
    : public gko::EnableLinOp<NoTransposableOperator> {
    NoTransposableOperator(std::shared_ptr<const gko::Executor> exec,
                           gko::dim<2> size = {})
        : gko::EnableLinOp<NoTransposableOperator>(exec, size)
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override {}

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}
};


class Transposition : public ::testing::Test {
protected:
    Transposition() : exec{gko::ReferenceExecutor::create()}
    {
        global_step = 0;
        op = std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 1});
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<gko::LinOp> op;
};


TEST_F(Transposition, CanBeEmpty)
{
    auto trans = gko::Transposition::create(exec);

    ASSERT_EQ(trans->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(trans->get_operator(), nullptr);
    ASSERT_EQ(trans->get_transposition(), nullptr);
}


TEST_F(Transposition, CanCreateFromOperator)
{
    auto trans = gko::Transposition::create(op);

    ASSERT_EQ(trans->get_size(), gko::dim<2>(1, 2));
    ASSERT_EQ(trans->get_operator()->get_size(), gko::dim<2>(2, 1));
}


TEST_F(Transposition, ThrowOpNotTransposable)
{
    auto no_trans_op =
        std::make_shared<NoTransposableOperator>(exec, gko::dim<2>(2, 1));

    ASSERT_THROW(gko::Transposition::create(no_trans_op), gko::NotSupported);
}


TEST_F(Transposition, ConstructNow)
{
    auto trans =
        gko::Transposition::create(op, gko::transposition::behavior::now);

    global_step++;
    trans->prepare_transposition();

    ASSERT_EQ(gko::as<DummyOperator>(trans->get_operator())->get_step(), 0);
    // the transposed operator is existed in the construction, so it used the
    // old global_step 0.
    ASSERT_EQ(gko::as<DummyOperator>(trans->get_transposition())->get_step(),
              0);
    ASSERT_EQ(trans->get_transposition()->get_size(), gko::dim<2>(1, 2));
}


TEST_F(Transposition, ConstructLazy)
{
    auto trans =
        gko::Transposition::create(op, gko::transposition::behavior::lazy);

    global_step++;
    trans->prepare_transposition();

    ASSERT_EQ(gko::as<DummyOperator>(trans->get_operator())->get_step(), 0);
    // the transposed operator is non-existed in the construction, so it used
    // the new global_step 1.
    ASSERT_EQ(gko::as<DummyOperator>(trans->get_transposition())->get_step(),
              1);
    ASSERT_EQ(trans->get_transposition()->get_size(), gko::dim<2>(1, 2));
}


TEST_F(Transposition, ConstructDefaultLazy)
{
    auto trans = gko::Transposition::create(op);

    global_step++;
    trans->prepare_transposition();

    ASSERT_EQ(gko::as<DummyOperator>(trans->get_operator())->get_step(), 0);
    // the transposed operator is non-existed in the construction, so it used
    // the new global_step 1.
    ASSERT_EQ(gko::as<DummyOperator>(trans->get_transposition())->get_step(),
              1);
    ASSERT_EQ(trans->get_transposition()->get_size(), gko::dim<2>(1, 2));
}


TEST_F(Transposition, LazyConstructOnlyOnce)
{
    auto trans = gko::Transposition::create(op);
    auto b = std::make_shared<DummyOperator>(exec, gko::dim<2>(2, 1));
    auto x = std::make_shared<DummyOperator>(exec, gko::dim<2>(1, 1));

    global_step++;
    trans->prepare_transposition();
    global_step++;
    trans->apply(gko::lend(b), gko::lend(x));
    global_step++;
    trans->prepare_transposition();

    ASSERT_EQ(gko::as<DummyOperator>(trans->get_transposition())->get_step(),
              1);
}


}  // namespace

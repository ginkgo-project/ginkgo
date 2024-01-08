// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/rcm.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


template <typename ValueIndexType>
class Rcm : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using reorder_type = gko::reorder::Rcm<v_type, i_type>;
    using Mtx = gko::matrix::Dense<v_type>;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    Rcm()
        : exec(gko::ReferenceExecutor::create()),
          rcm_factory(reorder_type::build().on(exec)),
          // clang-format off
          id3_mtx(gko::initialize<CsrMtx>(
              {{1.0, 0.0, 0.0},
              {0.0, 1.0, 0.0},
              {0.0, 0.0, 1.0}}, exec)),
          not_id3_mtx(gko::initialize<CsrMtx>(
              {{1.0, 0.0, 1.0},
              {0.0, 1.0, 0.0},
              {1.0, 0.0, 1.0}}, exec)),
          // clang-format on
          reorder_op(rcm_factory->generate(id3_mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<CsrMtx> id3_mtx;
    std::shared_ptr<CsrMtx> not_id3_mtx;
    std::unique_ptr<typename reorder_type::Factory> rcm_factory;
    std::unique_ptr<reorder_type> reorder_op;
};

TYPED_TEST_SUITE(Rcm, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Rcm, CanBeCleared)
{
    this->reorder_op->clear();

    auto reorder_op_perm = this->reorder_op->get_permutation();

    ASSERT_EQ(reorder_op_perm, nullptr);
}


TYPED_TEST(Rcm, CanBeCopied)
{
    auto rcm = this->rcm_factory->generate(this->id3_mtx);
    auto rcm_copy = this->rcm_factory->generate(this->not_id3_mtx);

    rcm_copy->copy_from(rcm);

    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[2], 0);
}


TYPED_TEST(Rcm, CanBeMoved)
{
    auto rcm = this->rcm_factory->generate(this->id3_mtx);
    auto rcm_move = this->rcm_factory->generate(this->not_id3_mtx);

    rcm->move_to(rcm_move);

    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[2], 0);
}


TYPED_TEST(Rcm, CanBeCloned)
{
    auto rcm = this->rcm_factory->generate(this->id3_mtx);

    auto rcm_clone = rcm->clone();

    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[2], 0);
}


TYPED_TEST(Rcm, HasSensibleDefaults)
{
    using reorder_type = typename TestFixture::reorder_type;

    auto rcm = reorder_type::build().on(this->exec)->generate(this->id3_mtx);

    ASSERT_EQ(rcm->get_parameters().construct_inverse_permutation, false);
    ASSERT_EQ(rcm->get_parameters().strategy,
              gko::reorder::starting_strategy::pseudo_peripheral);
}


TYPED_TEST(Rcm, CanBeCreatedWithStartingStrategy)
{
    using v_type = typename TestFixture::v_type;
    using reorder_type = typename TestFixture::reorder_type;
    auto rcm =
        reorder_type::build()
            .with_strategy(gko::reorder::starting_strategy::minimum_degree)
            .on(this->exec)
            ->generate(gko::initialize<gko::matrix::Dense<v_type>>(
                3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                this->exec));

    ASSERT_EQ(rcm->get_parameters().strategy,
              gko::reorder::starting_strategy::minimum_degree);
}


TYPED_TEST(Rcm, CanBeCreatedWithConstructInversePermutation)
{
    using reorder_type = typename TestFixture::reorder_type;
    auto rcm = reorder_type::build()
                   .with_construct_inverse_permutation(true)
                   .on(this->exec)
                   ->generate(this->id3_mtx);

    ASSERT_EQ(rcm->get_inverse_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm->get_inverse_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm->get_inverse_permutation()->get_const_permutation()[2], 0);
    ASSERT_EQ(rcm->get_parameters().construct_inverse_permutation, true);
}


}  // namespace

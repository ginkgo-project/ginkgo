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

#include <ginkgo/core/reorder/mc64.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


template <typename ValueIndexType>
class Mc64 : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<v_type>;
    using reorder_type = gko::reorder::Mc64<v_type, i_type>;
    using Mtx = gko::matrix::Dense<v_type>;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    Mc64()
        : exec(gko::ReferenceExecutor::create()),
          mc64_factory(reorder_type::build().on(exec)),
          // clang-format off
          id3_mtx(gko::initialize<CsrMtx>(
              {{1.0, 0.0, 0.0}, 
              {0.0, 1.0, 0.0}, 
              {0.0, 0.0, 1.0}}, exec)),
          not_id3_mtx(gko::initialize<CsrMtx>(
              {{1.0, 0.0, 2.0}, 
              {0.0, 1.0, 0.0}, 
              {2.0, 0.0, 1.0}}, exec)),
          // clang-format on
          reorder_op(mc64_factory->generate(id3_mtx))
    {}

    void assert_correct_permutation(const reorder_type* mc64)
    {
        auto perm =
            gko::as<gko::matrix::Permutation<i_type>>(mc64->get_permutation())
                ->get_const_permutation();

        ASSERT_EQ(perm[0], 0);
        ASSERT_EQ(perm[1], 1);
        ASSERT_EQ(perm[2], 2);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<CsrMtx> id3_mtx;
    std::shared_ptr<CsrMtx> not_id3_mtx;
    std::unique_ptr<typename reorder_type::Factory> mc64_factory;
    std::unique_ptr<reorder_type> reorder_op;
};

TYPED_TEST_SUITE(Mc64, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Mc64, CanBeCleared)
{
    this->reorder_op->clear();

    auto reorder_op_perm = this->reorder_op->get_permutation();

    ASSERT_EQ(reorder_op_perm, nullptr);
}


TYPED_TEST(Mc64, CanBeCopied)
{
    auto mc64 = this->mc64_factory->generate(this->id3_mtx);
    auto mc64_copy = this->mc64_factory->generate(this->not_id3_mtx);

    mc64_copy->copy_from(mc64.get());

    this->assert_correct_permutation(mc64_copy.get());
}


TYPED_TEST(Mc64, CanBeMoved)
{
    auto mc64 = this->mc64_factory->generate(this->id3_mtx);
    auto mc64_move = this->mc64_factory->generate(this->not_id3_mtx);

    mc64->move_to(mc64_move.get());

    this->assert_correct_permutation(mc64_move.get());
}


TYPED_TEST(Mc64, CanBeCloned)
{
    auto mc64 = this->mc64_factory->generate(this->id3_mtx);

    auto mc64_clone = mc64->clone();

    this->assert_correct_permutation(mc64_clone.get());
}


TYPED_TEST(Mc64, HasSensibleDefaults)
{
    using reorder_type = typename TestFixture::reorder_type;
    using real_type = typename TestFixture::real_type;

    auto mc64 = reorder_type::build().on(this->exec)->generate(this->id3_mtx);

    ASSERT_EQ(mc64->get_parameters().strategy,
              gko::reorder::reordering_strategy::max_diagonal_product);
    ASSERT_EQ(mc64->get_parameters().tolerance, real_type{1e-14});
    ASSERT_EQ(mc64->get_parameters().log2_degree, 4);
}


TYPED_TEST(Mc64, CanBeCreatedWithReorderingStrategy)
{
    using reorder_type = typename TestFixture::reorder_type;

    auto mc64 =
        reorder_type::build()
            .with_strategy(gko::reorder::reordering_strategy::max_diagonal_sum)
            .on(this->exec)
            ->generate(this->id3_mtx);

    this->assert_correct_permutation(mc64.get());
}


TYPED_TEST(Mc64, CanBeCreatedWithTolerance)
{
    using reorder_type = typename TestFixture::reorder_type;
    using real_type = typename TestFixture::real_type;

    auto mc64 = reorder_type::build()
                    .with_tolerance(real_type{1e-10})
                    .on(this->exec)
                    ->generate(this->id3_mtx);

    this->assert_correct_permutation(mc64.get());
}


TYPED_TEST(Mc64, CanBeCreatedWithLog2Degree)
{
    using reorder_type = typename TestFixture::reorder_type;

    auto mc64 = reorder_type::build()
                    .with_log2_degree(2)
                    .on(this->exec)
                    ->generate(this->id3_mtx);

    this->assert_correct_permutation(mc64.get());
}


}  // namespace

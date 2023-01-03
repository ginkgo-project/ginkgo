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

#include <ginkgo/core/matrix/row_gatherer.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


template <typename ValueIndexType>
class RowGatherer : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using o_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<2, decltype(ValueIndexType())>::type;
    using Vec = gko::matrix::Dense<v_type>;
    using OutVec = gko::matrix::Dense<o_type>;
    RowGatherer()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::RowGatherer<i_type>::create(
              exec, gko::dim<2>{4, 3}, gko::array<i_type>{exec, {1, 0, 2, 1}})),
          in(gko::initialize<Vec>(
              {{1.0, -1.0, 3.0}, {0.0, -2.0, 1.0}, {2.0, 0.0, -2.0}}, exec)),
          out(gko::initialize<OutVec>({{0.0, -1.0, 1.0},
                                       {1.0, -1.0, 1.0},
                                       {-1.0, 0.0, -1.0},
                                       {1.0, -1.0, 3.0}},
                                      exec))
    {}


    static void assert_equal_to_original_mtx(
        const gko::matrix::RowGatherer<i_type>* m)
    {
        auto gather = m->get_const_row_idxs();
        ASSERT_EQ(m->get_size(), gko::dim<2>(4, 3));
        ASSERT_EQ(gather[0], 1);
        ASSERT_EQ(gather[1], 0);
        ASSERT_EQ(gather[2], 2);
        ASSERT_EQ(gather[3], 1);
    }

    static void assert_empty(gko::matrix::RowGatherer<i_type>* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_const_row_idxs(), nullptr);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::RowGatherer<i_type>> mtx;
    std::unique_ptr<Vec> in;
    std::unique_ptr<OutVec> out;
};

TYPED_TEST_SUITE(RowGatherer, gko::test::TwoValueIndexType,
                 TupleTypenameNameGenerator);


TYPED_TEST(RowGatherer, CanBeEmpty)
{
    using i_type = typename TestFixture::i_type;
    auto empty = gko::matrix::RowGatherer<i_type>::create(this->exec);

    this->assert_empty(empty.get());
}


TYPED_TEST(RowGatherer, CanBeConstructedWithSize)
{
    using i_type = typename TestFixture::i_type;
    auto m =
        gko::matrix::RowGatherer<i_type>::create(this->exec, gko::dim<2>{2, 3});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
}


TYPED_TEST(RowGatherer, RowGathererCanBeConstructedFromExistingData)
{
    using i_type = typename TestFixture::i_type;
    i_type data[] = {1, 0, 2};

    auto m = gko::matrix::RowGatherer<i_type>::create(
        this->exec, gko::dim<2>{3, 5},
        gko::make_array_view(this->exec, 3, data));

    ASSERT_EQ(m->get_const_row_idxs(), data);
}


TYPED_TEST(RowGatherer, RowGathererThrowsforWrongRowPermDimensions)
{
    using i_type = typename TestFixture::i_type;
    i_type data[] = {0, 2, 1};

    ASSERT_THROW(gko::matrix::RowGatherer<i_type>::create(
                     this->exec, gko::dim<2>{4, 2},
                     gko::make_array_view(this->exec, 3, data)),
                 gko::ValueMismatch);
}


TYPED_TEST(RowGatherer, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(RowGatherer, CanBeCreatedFromExistingConstData)
{
    using i_type = typename TestFixture::i_type;
    const i_type row_idxs[] = {1, 0, 2, 1};

    auto const_mtx = gko::matrix::RowGatherer<i_type>::create_const(
        this->exec, gko::dim<2>{4, 3},
        gko::array<i_type>::const_view(this->exec, 4, row_idxs));

    this->assert_equal_to_original_mtx(const_mtx.get());
}


TYPED_TEST(RowGatherer, CanBeCopied)
{
    using i_type = typename TestFixture::i_type;
    auto mtx_copy = gko::matrix::RowGatherer<i_type>::create(this->exec);

    mtx_copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_row_idxs()[0] = 3;
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(RowGatherer, CanBeMoved)
{
    using i_type = typename TestFixture::i_type;
    auto mtx_copy = gko::matrix::RowGatherer<i_type>::create(this->exec);

    mtx_copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(RowGatherer, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(RowGatherer, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(RowGatherer, CanRowGatherMixed)
{
    using o_type = typename TestFixture::o_type;
    this->mtx->apply(this->in.get(), this->out.get());

    GKO_ASSERT_MTX_NEAR(this->out,
                        l<o_type>({{0.0, -2.0, 1.0},
                                   {1.0, -1.0, 3.0},
                                   {2.0, 0.0, -2.0},
                                   {0.0, -2.0, 1.0}}),
                        0.0);
}


TYPED_TEST(RowGatherer, CanAdvancedRowGatherMixed)
{
    using o_type = typename TestFixture::o_type;
    using v_type = typename TestFixture::v_type;
    using Vec = typename TestFixture::Vec;
    using OutVec = typename TestFixture::OutVec;
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);

    this->mtx->apply(alpha.get(), this->in.get(), beta.get(), this->out.get());

    GKO_ASSERT_MTX_NEAR(this->out,
                        l<o_type>({{0.0, -3.0, 1.0},
                                   {1.0, -1.0, 5.0},
                                   {5.0, 0.0, -3.0},
                                   {-1.0, -3.0, -1.0}}),
                        0.0);
}


}  // namespace

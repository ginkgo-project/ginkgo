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

#include <ginkgo/core/matrix/permutation.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Permutation : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Vec = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    Permutation()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Permutation<index_type>::create(
              exec, gko::dim<2>{4, 3},
              gko::array<index_type>{exec, {1, 0, 2, 3}}))
    {}


    static void assert_equal_to_original_mtx(
        gko::ptr_param<gko::matrix::Permutation<index_type>> m)
    {
        auto perm = m->get_permutation();
        ASSERT_EQ(m->get_size(), gko::dim<2>(4, 3));
        ASSERT_EQ(m->get_size()[0], 4);
        ASSERT_EQ(perm[0], 1);
        ASSERT_EQ(perm[1], 0);
        ASSERT_EQ(perm[2], 2);
        ASSERT_EQ(perm[3], 3);
    }

    static void assert_empty(gko::matrix::Permutation<index_type>* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_size()[0], 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Permutation<index_type>> mtx;
};

TYPED_TEST_SUITE(Permutation, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Permutation, CanBeEmpty)
{
    using index_type = typename TestFixture::index_type;
    auto empty = gko::matrix::Permutation<index_type>::create(this->exec);

    this->assert_empty(empty.get());
}


TYPED_TEST(Permutation, ReturnsNullValuesArrayWhenEmpty)
{
    using index_type = typename TestFixture::index_type;
    auto empty = gko::matrix::Permutation<index_type>::create(this->exec);

    ASSERT_EQ(empty->get_const_permutation(), nullptr);
}


TYPED_TEST(Permutation, CanBeConstructedWithSize)
{
    using index_type = typename TestFixture::index_type;
    auto m = gko::matrix::Permutation<index_type>::create(this->exec,
                                                          gko::dim<2>{2, 3});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_size()[0], 2);
}


TYPED_TEST(Permutation, FactorySetsCorrectPermuteMask)
{
    using index_type = typename TestFixture::index_type;
    auto m = gko::matrix::Permutation<index_type>::create(this->exec);
    auto mask = m->get_permute_mask();

    ASSERT_EQ(mask, gko::matrix::row_permute);
}


TYPED_TEST(Permutation, PermutationCanBeConstructedFromExistingData)
{
    using index_type = typename TestFixture::index_type;
    index_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<index_type>::create(
        this->exec, gko::dim<2>{3, 5},
        gko::make_array_view(this->exec, 3, data));

    ASSERT_EQ(m->get_const_permutation(), data);
}


TYPED_TEST(Permutation, PermutationCanBeConstructedFromExistingConstData)
{
    using index_type = typename TestFixture::index_type;
    using index_type = typename TestFixture::index_type;
    const index_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<index_type>::create_const(
        this->exec, 3, gko::array<index_type>::const_view(this->exec, 3, data));

    ASSERT_EQ(m->get_const_permutation(), data);
}


TYPED_TEST(Permutation, CanBeConstructedWithSizeAndMask)
{
    using index_type = typename TestFixture::index_type;
    auto m = gko::matrix::Permutation<index_type>::create(
        this->exec, gko::dim<2>{2, 3}, gko::matrix::column_permute);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_size()[0], 2);
    ASSERT_EQ(m->get_permute_mask(), gko::matrix::column_permute);
}


TYPED_TEST(Permutation, CanExplicitlyOverrideSetPermuteMask)
{
    using index_type = typename TestFixture::index_type;
    auto m = gko::matrix::Permutation<index_type>::create(
        this->exec, gko::dim<2>{2, 3}, gko::matrix::column_permute);

    auto mask = m->get_permute_mask();
    ASSERT_EQ(mask, gko::matrix::column_permute);

    m->set_permute_mask(gko::matrix::row_permute |
                        gko::matrix::inverse_permute);

    auto s_mask = m->get_permute_mask();
    ASSERT_EQ(s_mask, gko::matrix::row_permute | gko::matrix::inverse_permute);
}


TYPED_TEST(Permutation, PermutationThrowsforWrongRowPermDimensions)
{
    using index_type = typename TestFixture::index_type;
    index_type data[] = {0, 2, 1};

    ASSERT_THROW(gko::matrix::Permutation<index_type>::create(
                     this->exec, gko::dim<2>{4, 2},
                     gko::make_array_view(this->exec, 3, data)),
                 gko::ValueMismatch);
}


TYPED_TEST(Permutation, SettingMaskDoesNotModifyData)
{
    using index_type = typename TestFixture::index_type;
    index_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<index_type>::create(
        this->exec, gko::dim<2>{3, 5},
        gko::make_array_view(this->exec, 3, data));

    auto mask = m->get_permute_mask();
    ASSERT_EQ(m->get_const_permutation(), data);
    ASSERT_EQ(mask, gko::matrix::row_permute);

    m->set_permute_mask(gko::matrix::row_permute |
                        gko::matrix::inverse_permute);

    auto s_mask = m->get_permute_mask();
    ASSERT_EQ(s_mask, gko::matrix::row_permute | gko::matrix::inverse_permute);
    ASSERT_EQ(m->get_const_permutation(), data);
}


TYPED_TEST(Permutation, PermutationThrowsforWrongColPermDimensions)
{
    using index_type = typename TestFixture::index_type;
    index_type data[] = {0, 2, 1};

    ASSERT_THROW(gko::matrix::Permutation<index_type>::create(
                     this->exec, gko::dim<2>{3, 4},
                     gko::make_array_view(this->exec, 3, data),
                     gko::matrix::column_permute),
                 gko::ValueMismatch);
}


TYPED_TEST(Permutation, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx);
}


TYPED_TEST(Permutation, CanBeCopied)
{
    using index_type = typename TestFixture::index_type;
    auto mtx_copy = gko::matrix::Permutation<index_type>::create(this->exec);

    mtx_copy->copy_from(this->mtx);

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_permutation()[0] = 3;
    this->assert_equal_to_original_mtx(mtx_copy);
}


TYPED_TEST(Permutation, CanBeMoved)
{
    using index_type = typename TestFixture::index_type;
    auto mtx_copy = gko::matrix::Permutation<index_type>::create(this->exec);

    mtx_copy->move_from(this->mtx);

    this->assert_equal_to_original_mtx(mtx_copy);
}


TYPED_TEST(Permutation, CopyingPreservesMask)
{
    using index_type = typename TestFixture::index_type;
    auto mtx_copy = gko::matrix::Permutation<index_type>::create(this->exec);

    mtx_copy->copy_from(this->mtx);

    auto o_mask = this->mtx->get_permute_mask();
    auto n_mask = mtx_copy->get_permute_mask();
    ASSERT_EQ(o_mask, gko::matrix::row_permute);
    ASSERT_EQ(o_mask, n_mask);

    this->mtx->set_permute_mask(gko::matrix::column_permute);

    o_mask = this->mtx->get_permute_mask();
    n_mask = mtx_copy->get_permute_mask();
    ASSERT_EQ(o_mask, gko::matrix::column_permute);
    ASSERT_NE(o_mask, n_mask);

    mtx_copy->copy_from(this->mtx);

    n_mask = mtx_copy->get_permute_mask();
    ASSERT_EQ(o_mask, n_mask);
}


TYPED_TEST(Permutation, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(Permutation, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


}  // namespace

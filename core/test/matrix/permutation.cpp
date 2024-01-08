// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
              exec, gko::array<index_type>{exec, {1, 0, 2, 3}}))
    {}


    static void assert_equal_to_original_mtx(
        gko::ptr_param<gko::matrix::Permutation<index_type>> m)
    {
        auto perm = m->get_permutation();
        ASSERT_EQ(m->get_size(), gko::dim<2>(4, 4));
        ASSERT_EQ(perm[0], 1);
        ASSERT_EQ(perm[1], 0);
        ASSERT_EQ(perm[2], 2);
        ASSERT_EQ(perm[3], 3);
    }

    static void assert_empty(gko::matrix::Permutation<index_type>* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
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
    auto m = gko::matrix::Permutation<index_type>::create(this->exec, 2);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 2));
}


TYPED_TEST(Permutation, PermutationCanBeConstructedFromExistingData)
{
    using index_type = typename TestFixture::index_type;
    index_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<index_type>::create(
        this->exec, gko::make_array_view(this->exec, 3, data));

    ASSERT_EQ(m->get_const_permutation(), data);
}


TYPED_TEST(Permutation, PermutationCanBeConstructedFromExistingConstData)
{
    using index_type = typename TestFixture::index_type;
    using index_type = typename TestFixture::index_type;
    const index_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<index_type>::create_const(
        this->exec, gko::array<index_type>::const_view(this->exec, 3, data));

    ASSERT_EQ(m->get_const_permutation(), data);
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

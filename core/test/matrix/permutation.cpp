/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include "core/test/utils/assertions.hpp"


namespace {


class Permutation : public ::testing::Test {
protected:
    using i_type = int;
    using v_type = double;
    using Vec = gko::matrix::Dense<v_type>;
    using Csr = gko::matrix::Csr<v_type, i_type>;
    Permutation()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Permutation<i_type>::create(
              exec, gko::dim<2>{4, 3}, gko::Array<i_type>{exec, {1, 0, 2, 3}}))
    {}


    static void assert_equal_to_original_mtx(
        gko::matrix::Permutation<i_type> *m)
    {
        auto perm = m->get_permutation();
        ASSERT_EQ(m->get_size(), gko::dim<2>(4, 3));
        ASSERT_EQ(m->get_permutation_size(), 4);
        ASSERT_EQ(perm[0], 1);
        ASSERT_EQ(perm[1], 0);
        ASSERT_EQ(perm[2], 2);
        ASSERT_EQ(perm[3], 3);
    }

    static void assert_empty(gko::matrix::Permutation<i_type> *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_permutation_size(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Permutation<i_type>> mtx;
};


TEST_F(Permutation, CanBeEmpty)
{
    auto empty = gko::matrix::Permutation<i_type>::create(exec);

    assert_empty(empty.get());
}


TEST_F(Permutation, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::Permutation<i_type>::create(exec);

    ASSERT_EQ(empty->get_const_permutation(), nullptr);
}


TEST_F(Permutation, CanBeConstructedWithSize)
{
    auto m = gko::matrix::Permutation<i_type>::create(exec, gko::dim<2>{2, 3});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_permutation_size(), 2);
}


TEST_F(Permutation, PermutationCanBeConstructedFromExistingData)
{
    i_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3, 5}, gko::Array<i_type>::view(exec, 3, data));

    ASSERT_EQ(m->get_const_permutation(), data);
}


TEST_F(Permutation, PermutationThrowsforWrongRowPermDimensions)
{
    i_type data[] = {0, 2, 1};

    ASSERT_THROW(
        gko::matrix::Permutation<>::create(
            exec, gko::dim<2>{4, 2}, gko::Array<i_type>::view(exec, 3, data)),
        gko::ValueMismatch);
}


TEST_F(Permutation, PermutationThrowsforWrongColPermDimensions)
{
    i_type data[] = {0, 2, 1};

    ASSERT_THROW(
        gko::matrix::Permutation<>::create(
            exec, gko::dim<2>{3, 4}, gko::Array<i_type>::view(exec, 3, data),
            gko::matrix::column_permute),
        gko::ValueMismatch);
}


TEST_F(Permutation, KnowsItsSizeAndValues)
{
    assert_equal_to_original_mtx(mtx.get());
}


TEST_F(Permutation, CanBeCopied)
{
    auto mtx_copy = gko::matrix::Permutation<i_type>::create(exec);

    mtx_copy->copy_from(mtx.get());

    assert_equal_to_original_mtx(mtx.get());

    mtx->get_permutation()[0] = 3;

    assert_equal_to_original_mtx(mtx_copy.get());
}


TEST_F(Permutation, CanBeMoved)
{
    auto mtx_copy = gko::matrix::Permutation<i_type>::create(exec);

    mtx_copy->copy_from(std::move(mtx));

    assert_equal_to_original_mtx(mtx_copy.get());
}


TEST_F(Permutation, CanBeCloned)
{
    auto mtx_clone = mtx->clone();

    assert_equal_to_original_mtx(
        dynamic_cast<decltype(mtx.get())>(mtx_clone.get()));
}


TEST_F(Permutation, CanBeCleared)
{
    mtx->clear();

    assert_empty(mtx.get());
}


}  // namespace

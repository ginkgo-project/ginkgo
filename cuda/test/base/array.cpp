// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/array.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/base/array_access.hpp"
#include "cuda/test/utils.hpp"


template <typename T>
class Array : public CudaTestFixture {
protected:
    Array() : x(ref, 2)
    {
        x.get_data()[0] = 5;
        x.get_data()[1] = 2;
    }

    static void assert_equal_to_original_x(gko::array<T>& a)
    {
        ASSERT_EQ(a.get_size(), 2);
        EXPECT_EQ(a.get_data()[0], T{5});
        EXPECT_EQ(a.get_data()[1], T{2});
        EXPECT_EQ(a.get_const_data()[0], T{5});
        EXPECT_EQ(a.get_const_data()[1], T{2});
    }

    gko::array<T> x;
};

TYPED_TEST_SUITE(Array, gko::test::ValueAndIndexTypes, TypenameNameGenerator);


TYPED_TEST(Array, CanCreateTemporaryCloneOnDifferentExecutor)
{
    auto tmp_clone = make_temporary_clone(this->exec, &this->x);

    ASSERT_NE(tmp_clone.get(), &this->x);
    tmp_clone->set_executor(this->ref);
    this->assert_equal_to_original_x(*tmp_clone.get());
}


TYPED_TEST(Array, CanCopyBackTemporaryCloneOnDifferentExecutor)
{
    {
        auto tmp_clone = make_temporary_clone(this->exec, &this->x);
        // change x, so it no longer matches the original x
        // the copy-back will overwrite it again with the correct value
        this->x.get_data()[0] = 0;
    }

    this->assert_equal_to_original_x(this->x);
}


TYPED_TEST(Array, CanGetValue)
{
    using T = TypeParam;
    auto arr = gko::array<T>(this->exec, I<T>{4, 6});

    ASSERT_EQ(get_element(arr, 0), T{4});
    ASSERT_EQ(get_element(arr, 1), T{6});
}


TYPED_TEST(Array, CanSetValue)
{
    using T = TypeParam;
    auto arr = gko::array<T>(this->exec, I<T>{4, 6});

    set_element(arr, 1, 0);

    ASSERT_EQ(get_element(arr, 1), T{0});
}


TYPED_TEST(Array, CanBeReduced)
{
    using T = TypeParam;
    auto arr = gko::array<T>(this->exec, I<T>{4, 6});
    auto out = gko::array<T>(this->exec, I<T>{2});

    gko::reduce_add(arr, out);

    out.set_executor(this->exec->get_master());
    ASSERT_EQ(out.get_data()[0], T{12});
}


TYPED_TEST(Array, CanBeReduced2)
{
    using T = TypeParam;
    auto arr = gko::array<T>(this->exec, I<T>{4, 6});

    auto out = gko::reduce_add(arr, T{3});

    ASSERT_EQ(out, T{13});
}

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

#include <ginkgo/core/base/index_set.hpp>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class IndexSet : public ::testing::Test {
protected:
    using value_type = T;
    IndexSet() : exec(gko::ReferenceExecutor::create()) {}

    void TearDown()
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    static void assert_equal_to_original(gko::IndexSet<T> &a)
    {
        ASSERT_EQ(a.get_size(), 10);
    }

    static void assert_equal_index_sets(gko::IndexSet<T> &a,
                                        gko::IndexSet<T> &b)
    {
        ASSERT_EQ(a.get_size(), b.get_size());
        ASSERT_EQ(a.get_num_subsets(), b.get_num_subsets());
        if (a.get_num_subsets() > 0) {
            for (auto i = 0; i < a.get_num_subsets(); ++i) {
                ASSERT_EQ(a.get_subsets_begin()[i], b.get_subsets_begin()[i]);
                ASSERT_EQ(a.get_subsets_end()[i], b.get_subsets_end()[i]);
                ASSERT_EQ(a.get_superset_indices()[i],
                          b.get_superset_indices()[i]);
            }
        }
    }


    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(IndexSet, gko::test::IndexTypes);


TYPED_TEST(IndexSet, CanBeEmpty)
{
    auto empty = gko::IndexSet<TypeParam>{};
    ASSERT_EQ(empty.get_size(), 0);
    ASSERT_EQ(empty.get_num_subsets(), 0);
}


TYPED_TEST(IndexSet, CanBeConstructedWithSize)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set.get_num_subsets(), 0);
}


TYPED_TEST(IndexSet, CanBeCopyConstructed)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2(idx_set);
    this->assert_equal_index_sets(idx_set2, idx_set);
}


TYPED_TEST(IndexSet, CanBeMoveConstructed)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2(std::move(idx_set));
    this->assert_equal_to_original(idx_set2);
}


TYPED_TEST(IndexSet, CanBeCopyAssigned)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2 = idx_set;
    this->assert_equal_index_sets(idx_set2, idx_set);
}


TYPED_TEST(IndexSet, CanBeMoveAssigned)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2 = std::move(idx_set);
    this->assert_equal_to_original(idx_set2);
}


TYPED_TEST(IndexSet, KnowsItsSize)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    ASSERT_EQ(idx_set.get_size(), 10);
}


}  // namespace

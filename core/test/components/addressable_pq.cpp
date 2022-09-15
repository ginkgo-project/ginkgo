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

#include "core/components/addressable_pq.hpp"


#include <algorithm>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class AddressablePriorityQueue : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using pq_type = gko::addressable_priority_queue<value_type, index_type>;

    AddressablePriorityQueue() : exec(gko::ReferenceExecutor::create()) {}

    void assert_min(pq_type pq, value_type key, index_type val)
    {
        ASSERT_EQ(pq.min_key(), key);
        ASSERT_EQ(pq.min_val(), val);
        ASSERT_TRUE((pq.min() == std::pair<value_type, index_type>{key, val}));
        ASSERT_FALSE(pq.empty());
    }

    void test_pq_functionality(pq_type& pq)
    {
        pq.insert(value_type{.5}, 1);
        ASSERT_EQ(pq.size(), 1);
        assert_min(pq, .5, 1);

        // insert larger key
        const auto handle_7 = pq.insert(value_type{1.}, 7);
        ASSERT_EQ(pq.size(), 2);
        assert_min(pq, .5, 1);

        // insert min key
        const auto handle_4 = pq.insert(value_type{.1}, 4);
        ASSERT_EQ(pq.size(), 3);
        assert_min(pq, .1, 4);

        // update key to have different min
        pq.update_key(handle_4, value_type{.7});
        ASSERT_EQ(pq.size(), 3);
        assert_min(pq, .5, 1);

        // insert same key as min
        pq.insert(value_type{.5}, 2);
        ASSERT_EQ(pq.size(), 4);
        assert_min(pq, .5, 1);

        // update max to new min key
        pq.update_key(handle_7, value_type{.2});
        ASSERT_EQ(pq.size(), 4);
        assert_min(pq, .2, 7);

        // insert intermediate key
        pq.insert(value_type{.3}, 5);
        ASSERT_EQ(pq.size(), 5);
        assert_min(pq, .2, 7);

        // pop min works
        pq.pop_min();
        ASSERT_EQ(pq.size(), 4);
        assert_min(pq, .3, 5);

        // reset works
        pq.reset();
        ASSERT_EQ(pq.size(), 0);
        ASSERT_TRUE(pq.empty());
    }

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(AddressablePriorityQueue, gko::test::RealValueIndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(AddressablePriorityQueue, InitializesCorrectly)
{
    using pq_type = typename TestFixture::pq_type;
    pq_type pq{4};

    ASSERT_EQ(pq.size(), 0);
    ASSERT_TRUE(pq.empty());
}


TYPED_TEST(AddressablePriorityQueue, WorksWithDegree2)
{
    using pq_type = typename TestFixture::pq_type;
    pq_type pq{2};

    this->test_pq_functionality(pq);
}


TYPED_TEST(AddressablePriorityQueue, WorksWithDegree4)
{
    using pq_type = typename TestFixture::pq_type;
    pq_type pq{4};

    this->test_pq_functionality(pq);
}


}  // namespace

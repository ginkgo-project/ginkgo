// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    using pq_type2 = gko::addressable_priority_queue<value_type, index_type, 2>;
    using pq_type4 = gko::addressable_priority_queue<value_type, index_type, 4>;

    AddressablePriorityQueue() : exec(gko::ReferenceExecutor::create()) {}

    template <typename PQType>
    void assert_min(const PQType& pq, value_type key, index_type val)
    {
        ASSERT_FALSE(pq.empty());
        ASSERT_EQ(pq.min_key(), key);
        ASSERT_EQ(pq.min_node(), val);
        ASSERT_TRUE((pq.min() == std::pair<value_type, index_type>{key, val}));
    }

    template <typename PQType>
    void test_pq_functionality()
    {
        PQType pq{exec, 8};

        pq.insert(value_type{.5}, 1);
        ASSERT_EQ(pq.size(), 1);
        assert_min(pq, .5, 1);

        // insert larger key
        pq.insert(value_type{1.}, 7);
        ASSERT_EQ(pq.size(), 2);
        assert_min(pq, .5, 1);

        // insert min key
        pq.insert(value_type{.1}, 4);
        ASSERT_EQ(pq.size(), 3);
        assert_min(pq, .1, 4);

        // update key to have different min
        pq.update_key(value_type{.7}, 4);
        ASSERT_EQ(pq.size(), 3);
        assert_min(pq, .5, 1);

        // insert same key as min
        pq.insert(value_type{.5}, 2);
        ASSERT_EQ(pq.size(), 4);
        assert_min(pq, .5, 1);

        // update max to new min key
        pq.update_key(value_type{.2}, 7);
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
    using pq_type = typename TestFixture::pq_type2;
    pq_type pq{this->exec, 0};

    ASSERT_EQ(pq.size(), 0);
    ASSERT_TRUE(pq.empty());
}


TYPED_TEST(AddressablePriorityQueue, WorksWithDegree2)
{
    this->template test_pq_functionality<typename TestFixture::pq_type2>();
}


TYPED_TEST(AddressablePriorityQueue, WorksWithDegree4)
{
    this->template test_pq_functionality<typename TestFixture::pq_type4>();
}


}  // namespace

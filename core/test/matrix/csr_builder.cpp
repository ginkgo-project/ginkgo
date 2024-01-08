// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_builder.hpp"


#include <memory>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class CsrBuilder : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

protected:
    CsrBuilder()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(CsrBuilder, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(CsrBuilder, ReturnsCorrectArrays)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::matrix::CsrBuilder<value_type, index_type> builder{this->mtx};

    auto builder_col_idxs = builder.get_col_idx_array().get_data();
    auto builder_values = builder.get_value_array().get_data();
    auto ref_col_idxs = this->mtx->get_col_idxs();
    auto ref_values = this->mtx->get_values();

    ASSERT_EQ(builder_col_idxs, ref_col_idxs);
    ASSERT_EQ(builder_values, ref_values);
}


TYPED_TEST(CsrBuilder, UpdatesSrowOnDestruction)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    struct mock_strategy : public Mtx::strategy_type {
        virtual void process(const gko::array<index_type>&,
                             gko::array<index_type>*) override
        {
            *was_called = true;
        }

        virtual int64_t clac_size(const int64_t nnz) override { return 0; }

        virtual std::shared_ptr<typename Mtx::strategy_type> copy() override
        {
            return std::make_shared<mock_strategy>(*was_called);
        }

        mock_strategy(bool& flag) : Mtx::strategy_type(""), was_called(&flag) {}

        bool* was_called;
    };
    bool was_called{};
    this->mtx->set_strategy(std::make_shared<mock_strategy>(was_called));
    was_called = false;

    gko::matrix::CsrBuilder<value_type, index_type>{this->mtx};

    ASSERT_TRUE(was_called);
}


}  // namespace

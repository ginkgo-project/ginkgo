#include <core/matrix/dense.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>


namespace {


class Dense : public ::testing::Test {
protected:
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Dense<>::create(exec, 2, 3, 4))
    {
        auto vals = mtx->get_values();
        vals[0 * 4 + 0] = 1.0;
        vals[0 * 4 + 1] = 2.0;
        vals[0 * 4 + 2] = 3.0;
        vals[1 * 4 + 0] = 1.5;
        vals[1 * 4 + 1] = 2.5;
        vals[1 * 4 + 2] = 3.5;
    }


    static void assert_equal_to_original_mtx(gko::matrix::Dense<> *m)
    {
        ASSERT_EQ(m->get_num_rows(), 2);
        ASSERT_EQ(m->get_num_cols(), 3);
        ASSERT_EQ(m->get_padding(), 4);
        ASSERT_EQ(m->get_num_nonzeros(), 2 * 4);
        auto vals = m->get_const_values();
        EXPECT_EQ(vals[0 * 4 + 0], 1.0);
        EXPECT_EQ(vals[0 * 4 + 1], 2.0);
        EXPECT_EQ(vals[0 * 4 + 2], 3.0);
        EXPECT_EQ(vals[1 * 4 + 0], 1.5);
        EXPECT_EQ(vals[1 * 4 + 1], 2.5);
        ASSERT_EQ(vals[1 * 4 + 2], 3.5);
    }

    static void assert_empty(gko::matrix::Dense<> *m)
    {
        EXPECT_EQ(m->get_num_rows(), 0);
        EXPECT_EQ(m->get_num_cols(), 0);
        ASSERT_EQ(m->get_num_nonzeros(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Dense<>> mtx;
};


TEST_F(Dense, CanBeEmpty)
{
    auto empty = gko::matrix::Dense<>::create(exec);
    assert_empty(empty.get());
}


TEST_F(Dense, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::Dense<>::create(exec);
    ASSERT_EQ(empty->get_values(), nullptr);
}


TEST_F(Dense, KnowsItsSizeAndValues)
{
    assert_equal_to_original_mtx(mtx.get());
}


TEST_F(Dense, CanBeCopied)
{
    auto mtx_copy = gko::matrix::Dense<>::create(exec);
    mtx_copy->copy_from(mtx.get());
    assert_equal_to_original_mtx(mtx.get());
    mtx->get_values()[0] = 7;
    assert_equal_to_original_mtx(mtx_copy.get());
}


TEST_F(Dense, CanBeMoved)
{
    auto mtx_copy = gko::matrix::Dense<>::create(exec);
    mtx_copy->copy_from(std::move(mtx));
    assert_equal_to_original_mtx(mtx_copy.get());
}


TEST_F(Dense, CanBeCloned)
{
    auto mtx_clone = mtx->clone();
    assert_equal_to_original_mtx(
        dynamic_cast<decltype(mtx.get())>(mtx_clone.get()));
}


TEST_F(Dense, CanBeCleared)
{
    mtx->clear();
    assert_empty(mtx.get());
}


}  // namespace

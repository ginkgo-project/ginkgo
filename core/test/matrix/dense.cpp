#include <core/matrix/dense.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>


namespace {


class Dense : public ::testing::Test {
protected:
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Dense<>::create(exec, 4,
                                           {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}))
    {}


    static void assert_equal_to_original_mtx(gko::matrix::Dense<> *m)
    {
        ASSERT_EQ(m->get_num_rows(), 2);
        ASSERT_EQ(m->get_num_cols(), 3);
        ASSERT_EQ(m->get_padding(), 4);
        ASSERT_EQ(m->get_num_stored_elements(), 2 * 4);
        EXPECT_EQ(m->at(0, 0), 1.0);
        EXPECT_EQ(m->at(0, 1), 2.0);
        EXPECT_EQ(m->at(0, 2), 3.0);
        EXPECT_EQ(m->at(1, 0), 1.5);
        EXPECT_EQ(m->at(1, 1), 2.5);
        ASSERT_EQ(m->at(1, 2), 3.5);
    }

    static void assert_empty(gko::matrix::Dense<> *m)
    {
        EXPECT_EQ(m->get_num_rows(), 0);
        EXPECT_EQ(m->get_num_cols(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
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
    ASSERT_EQ(empty->get_values().get_data(), nullptr);
}


TEST_F(Dense, KnowsItsSizeAndValues)
{
    assert_equal_to_original_mtx(mtx.get());
}


TEST_F(Dense, CanBeListConstructed)
{
    auto m = gko::matrix::Dense<>::create(exec, {1.0, 2.0});

    EXPECT_EQ(m->get_num_rows(), 2);
    EXPECT_EQ(m->get_num_cols(), 1);
    ASSERT_EQ(m->get_num_stored_elements(), 2);
    EXPECT_EQ(m->at(0), 1);
    EXPECT_EQ(m->at(1), 2);
}


TEST_F(Dense, CanBeListConstructedWithPadding)
{
    auto m = gko::matrix::Dense<>::create(exec, 2, {1.0, 2.0});
    EXPECT_EQ(m->get_num_rows(), 2);
    EXPECT_EQ(m->get_num_cols(), 1);
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0), 1.0);
    EXPECT_EQ(m->at(1), 2.0);
}


TEST_F(Dense, CanBeDoubleListConstructed)
{
    auto m = gko::matrix::Dense<>::create(exec,
                                          {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

    EXPECT_EQ(m->get_num_rows(), 3);
    EXPECT_EQ(m->get_num_cols(), 2);
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0), 1.0);
    EXPECT_EQ(m->at(1), 2.0);
    EXPECT_EQ(m->at(2), 3.0);
    ASSERT_EQ(m->at(3), 4.0);
    EXPECT_EQ(m->at(4), 5.0);
}


TEST_F(Dense, CanBeDoubleListConstructedWithPadding)
{
    auto m = gko::matrix::Dense<>::create(exec, 4,
                                          {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

    EXPECT_EQ(m->get_num_rows(), 3);
    EXPECT_EQ(m->get_num_cols(), 2);
    ASSERT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), 1.0);
    EXPECT_EQ(m->at(1), 2.0);
    EXPECT_EQ(m->at(2), 3.0);
    ASSERT_EQ(m->at(3), 4.0);
    EXPECT_EQ(m->at(4), 5.0);
}


TEST_F(Dense, CanBeCopied)
{
    auto mtx_copy = gko::matrix::Dense<>::create(exec);
    mtx_copy->copy_from(mtx.get());
    assert_equal_to_original_mtx(mtx.get());
    mtx->at(0) = 7;
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

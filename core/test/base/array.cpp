#include <core/base/array.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>


namespace {


class Array : public ::testing::Test {
protected:
    Array() : exec(gko::ReferenceExecutor::create()), x(exec, 2)
    {
        x.get_data()[0] = 5;
        x.get_data()[1] = 2;
    }

    static void assert_equal_to_original_x(gko::Array<int> &a)
    {
        ASSERT_EQ(a.get_num_elems(), 2);
        EXPECT_EQ(a.get_data()[0], 5);
        EXPECT_EQ(a.get_data()[1], 2);
        EXPECT_EQ(a.get_const_data()[0], 5);
        EXPECT_EQ(a.get_const_data()[1], 2);
    }

    std::shared_ptr<const gko::Executor> exec;
    gko::Array<int> x;
};


TEST_F(Array, CanBeEmpty)
{
    gko::Array<int> a(exec);

    ASSERT_EQ(a.get_num_elems(), 0);
}


TEST_F(Array, ReturnsNullWhenEmpty)
{
    gko::Array<int> a(exec);

    EXPECT_EQ(a.get_const_data(), nullptr);
    ASSERT_EQ(a.get_data(), nullptr);
}


TEST_F(Array, KnowsItsSize) { ASSERT_EQ(x.get_num_elems(), 2); }


TEST_F(Array, ReturnsValidDataPtr)
{
    EXPECT_EQ(x.get_data()[0], 5);
    EXPECT_EQ(x.get_data()[1], 2);
}


TEST_F(Array, ReturnsValidConstDataPtr)
{
    EXPECT_EQ(x.get_const_data()[0], 5);
    EXPECT_EQ(x.get_const_data()[1], 2);
}


TEST_F(Array, KnowsItsExecutor) { ASSERT_EQ(x.get_executor(), exec); }


TEST_F(Array, CanBeCopyConstructed)
{
    gko::Array<int> a(x);
    x.get_data()[0] = 7;

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeMoveConstructed)
{
    gko::Array<int> a(std::move(x));

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeCopied)
{
    auto cpu = gko::CpuExecutor::create();
    gko::Array<int> a(cpu, 2);
    a = x;
    x.get_data()[0] = 7;

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeMoved)
{
    auto cpu = gko::CpuExecutor::create();
    gko::Array<int> a(cpu, 2);
    a = std::move(x);

    assert_equal_to_original_x(a);
}


TEST_F(Array, CanBeCleared)
{
    x.clear();

    ASSERT_EQ(x.get_num_elems(), 0);
    ASSERT_EQ(x.get_data(), nullptr);
    ASSERT_EQ(x.get_const_data(), nullptr);
}


TEST_F(Array, CanBeResized)
{
    x.resize(3);

    x.get_data()[0] = 1;
    x.get_data()[1] = 8;
    x.get_data()[2] = 7;

    EXPECT_EQ(x.get_const_data()[0], 1);
    EXPECT_EQ(x.get_const_data()[1], 8);
    EXPECT_EQ(x.get_const_data()[2], 7);
}


TEST_F(Array, ManagesExternalData)
{
    int *data = nullptr;
    ASSERT_NE(data = reinterpret_cast<int *>(std::malloc(3 * sizeof(int))),
              nullptr);
    data[0] = 1;
    data[1] = 8;
    data[2] = 7;
    x.manage(3, data);

    EXPECT_EQ(x.get_const_data()[0], 1);
    EXPECT_EQ(x.get_const_data()[1], 8);
    EXPECT_EQ(x.get_const_data()[2], 7);
}


TEST_F(Array, ReleasesData)
{
    int *data = x.get_data();

    ASSERT_NO_THROW(x.release());
    ASSERT_NO_THROW(std::free(data));
    ASSERT_EQ(x.get_data(), nullptr);
    ASSERT_EQ(x.get_const_data(), nullptr);
    ASSERT_EQ(x.get_num_elems(), 0);
}


TEST_F(Array, ChangesExecutors)
{
    auto cpu = gko::CpuExecutor::create();
    x.set_executor(cpu);

    ASSERT_EQ(x.get_executor(), cpu);
    assert_equal_to_original_x(x);
}


}  // namespace

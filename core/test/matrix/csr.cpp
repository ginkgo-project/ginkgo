#include <core/matrix/csr.hpp>


#include <gtest/gtest.h>


namespace {


class Csr : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Csr<>;

    Csr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Csr<>::create(exec, 2, 3, 4))
    {
        auto v = mtx->get_values().get_data();
        auto c = mtx->get_col_idxs().get_data();
        auto r = mtx->get_row_ptrs().get_data();
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        v[0] = 1.0;
        c[0] = 0;
        v[1] = 3.0;
        c[1] = 1;
        v[2] = 2.0;
        c[2] = 2;
        v[3] = 5.0;
        c[3] = 1;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_origianl_mtx(const Mtx *m)
    {
        auto v = m->get_values().get_const_data();
        auto c = m->get_col_idxs().get_const_data();
        auto r = m->get_row_ptrs().get_const_data();
        ASSERT_EQ(m->get_num_rows(), 2);
        ASSERT_EQ(m->get_num_cols(), 3);
        ASSERT_EQ(m->get_num_nonzeros(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 3);
        EXPECT_EQ(r[2], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 3.0);
        EXPECT_EQ(v[2], 2.0);
        EXPECT_EQ(v[3], 5.0);
    }
};


TEST_F(Csr, KnowsItsSize)
{
    ASSERT_EQ(mtx->get_num_rows(), 2);
    ASSERT_EQ(mtx->get_num_cols(), 3);
    ASSERT_EQ(mtx->get_num_nonzeros(), 4);
}


TEST_F(Csr, HasCorrectArraySizes)
{
    ASSERT_EQ(mtx->get_values().get_num_elems(), 4);
    ASSERT_EQ(mtx->get_col_idxs().get_num_elems(), 4);
    ASSERT_EQ(mtx->get_row_ptrs().get_num_elems(), 3);
}


TEST_F(Csr, CanBeEmpty)
{
    auto mtx = gko::matrix::Csr<>::create(exec);

    ASSERT_EQ(mtx->get_num_rows(), 0);
    ASSERT_EQ(mtx->get_num_cols(), 0);
    ASSERT_EQ(mtx->get_num_nonzeros(), 0);
    ASSERT_EQ(mtx->get_values().get_data(), nullptr);
    ASSERT_EQ(mtx->get_col_idxs().get_data(), nullptr);
    ASSERT_EQ(mtx->get_row_ptrs().get_data(), nullptr);
}


}  // namespace

#include <core/matrix/csr.hpp>

#include <gtest/gtest.h>

#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>

namespace {

class Csr : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;

    Csr()
        : exec(gko::GpuExecutor::create(0, gko::ReferenceExecutor::create())),
          mtx(Mtx::create(exec, 2, 3, 4))
    {
        Mtx::value_type *v = mtx->get_values();
        Mtx::index_type *c = mtx->get_col_idxs();
        Mtx::index_type *r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};

TEST_F(Csr, AppliesToDenseVector)
{
    auto x = Vec::create(exec, {2.0, 1.0, 4.0});
    auto y = Vec::create(exec, 2, 1, 1);

    mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0), 13.0);
    EXPECT_EQ(y->at(1), 5.0);
}

TEST_F(Csr, AppliesToDenseMatrix)
{
    auto x = Vec::create(exec, {{2.0, 3.0}, {1.0, -1.5}, {4.0, 2.5}});
    auto y = Vec::create(exec, 2, 2, 2);

    mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0), 13.0);
    EXPECT_EQ(y->at(1, 0), 5.0);
    EXPECT_EQ(y->at(0, 1), 3.5);
    EXPECT_EQ(y->at(1, 1), -7.5);
}

TEST_F(Csr, AppliesLinearCombinationToDenseVector)
{
    auto alpha = Vec::create(exec, {-1.0});
    auto beta = Vec::create(exec, {2.0});
    auto x = Vec::create(exec, {2.0, 1.0, 4.0});
    auto y = Vec::create(exec, {1.0, 2.0});

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0), -11.0);
    EXPECT_EQ(y->at(1), -1.0);
}

TEST_F(Csr, AppliesLinearCombinationToDenseMatrix)
{
    auto alpha = Vec::create(exec, {-1.0});
    auto beta = Vec::create(exec, {2.0});
    auto x = Vec::create(exec, {{2.0, 3.0}, {1.0, -1.5}, {4.0, 2.5}});
    auto y = Vec::create(exec, {{1.0, 0.5}, {2.0, -1.5}});

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0, 0), -11.0);
    EXPECT_EQ(y->at(1, 0), -1.0);
    EXPECT_EQ(y->at(0, 1), -2.5);
    EXPECT_EQ(y->at(1, 1), 4.5);
}

TEST_F(Csr, ApplyFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, 2, 2, 2);
    auto y = Vec::create(exec, 2, 2, 2);

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}

TEST_F(Csr, ApplyFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, 3, 2, 2);
    auto y = Vec::create(exec, 3, 2, 2);

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}

TEST_F(Csr, ApplyFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, 3, 3, 2);
    auto y = Vec::create(exec, 2, 2, 2);

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}

}  // namespace

#include <core/matrix/csr.hpp>


#include <gtest/gtest.h>


namespace {


class Csr : public ::testing::Test {
protected:
    Csr() {}
};


TEST_F(Csr, CanBeEmpty)
{
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::matrix::Csr<>::create(exec);

    ASSERT_EQ(mtx->get_num_rows(), 0);
    ASSERT_EQ(mtx->get_num_cols(), 0);
    ASSERT_EQ(mtx->get_num_nonzeros(), 0);
}


}  // namespace

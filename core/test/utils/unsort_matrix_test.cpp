// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/test/utils/unsort_matrix.hpp"


#include <cmath>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class UnsortMatrix : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    UnsortMatrix()
        : exec(gko::ReferenceExecutor::create()),
          rand_engine(42),
          csr_empty(Csr::create(exec, gko::dim<2>(0, 0))),
          coo_empty(Coo::create(exec, gko::dim<2>(0, 0)))
    {}
    /*
     Matrix used for both CSR and COO:
              1, 2, 0, 0, 0
              0, 0, 0, 0, 0
              3, 4, 5, 6, 0
              0, 0, 7, 0, 0
              0, 0, 8, 9, 10
     */
    std::unique_ptr<Csr> get_sorted_csr()
    {
        return Csr::create(exec, gko::dim<2>{5, 5},
                           I<value_type>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                           I<index_type>{0, 1, 0, 1, 2, 3, 2, 2, 3, 4},
                           I<index_type>{0, 2, 2, 6, 7, 10});
    }

    std::unique_ptr<Coo> get_sorted_coo()
    {
        return Coo::create(exec, gko::dim<2>{5, 5},
                           I<value_type>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                           I<index_type>{0, 1, 0, 1, 2, 3, 2, 2, 3, 4},
                           I<index_type>{0, 0, 2, 2, 2, 2, 3, 4, 4, 4});
    }

    bool is_coo_matrix_sorted(gko::ptr_param<Coo> mtx)
    {
        auto rows = mtx->get_const_row_idxs();
        auto cols = mtx->get_const_col_idxs();
        auto nnz = mtx->get_num_stored_elements();

        if (nnz <= 0) {
            return true;
        }

        auto prev_row = rows[0];
        auto prev_col = cols[0];
        for (index_type i = 0; i < nnz; ++i) {
            auto cur_row = rows[i];
            auto cur_col = cols[i];
            if (prev_row == cur_row && prev_col > cur_col) {
                return false;
            }
            prev_row = cur_row;
            prev_col = cur_col;
        }
        return true;
    }

    bool is_csr_matrix_sorted(gko::ptr_param<Csr> mtx)
    {
        auto size = mtx->get_size();
        auto rows = mtx->get_const_row_ptrs();
        auto cols = mtx->get_const_col_idxs();
        auto nnz = mtx->get_num_stored_elements();

        if (nnz <= 0) {
            return true;
        }

        for (index_type row = 0; row < size[1]; ++row) {
            auto prev_col = cols[rows[row]];
            for (index_type i = rows[row]; i < rows[row + 1]; ++i) {
                auto cur_col = cols[i];
                if (prev_col > cur_col) {
                    return false;
                }
                prev_col = cur_col;
            }
        }
        return true;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::default_random_engine rand_engine;
    std::unique_ptr<Csr> csr_empty;
    std::unique_ptr<Coo> coo_empty;
};

TYPED_TEST_SUITE(UnsortMatrix, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(UnsortMatrix, CsrWorks)
{
    auto csr = this->get_sorted_csr();
    const auto ref_mtx = this->get_sorted_csr();
    bool was_sorted = this->is_csr_matrix_sorted(csr);

    gko::test::unsort_matrix(csr, this->rand_engine);

    ASSERT_FALSE(this->is_csr_matrix_sorted(csr));
    ASSERT_TRUE(was_sorted);
    GKO_ASSERT_MTX_NEAR(csr, ref_mtx, 0.);
}


TYPED_TEST(UnsortMatrix, CsrWorksWithEmpty)
{
    const bool was_sorted = this->is_csr_matrix_sorted(this->csr_empty);

    gko::test::unsort_matrix(this->csr_empty, this->rand_engine);

    ASSERT_TRUE(was_sorted);
    ASSERT_EQ(this->csr_empty->get_num_stored_elements(), 0);
}


TYPED_TEST(UnsortMatrix, CooWorks)
{
    auto coo = this->get_sorted_coo();
    const auto ref_mtx = this->get_sorted_coo();
    const bool was_sorted = this->is_coo_matrix_sorted(coo);

    gko::test::unsort_matrix(coo, this->rand_engine);

    ASSERT_FALSE(this->is_coo_matrix_sorted(coo));
    ASSERT_TRUE(was_sorted);
    GKO_ASSERT_MTX_NEAR(coo, ref_mtx, 0.);
}


TYPED_TEST(UnsortMatrix, CooWorksWithEmpty)
{
    const bool was_sorted = this->is_coo_matrix_sorted(this->coo_empty);

    gko::test::unsort_matrix(this->coo_empty, this->rand_engine);

    ASSERT_TRUE(was_sorted);
    ASSERT_EQ(this->coo_empty->get_num_stored_elements(), 0);
}


}  // namespace

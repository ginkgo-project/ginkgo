// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sparsity_csr_kernels.hpp"


#include <algorithm>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "test/utils/executor.hpp"


namespace {


class SparsityCsr : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::SparsityCsr<value_type, index_type>;
    using Mtx64 = gko::matrix::SparsityCsr<value_type, gko::int64>;

    SparsityCsr() : rng{9312}
    {
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                100, 100, std::uniform_int_distribution<index_type>(1, 10),
                std::uniform_real_distribution<value_type>(0.0, 1.0), rng);
        // make sure the matrix contains a few diagonal entries
        for (int i = 0; i < 10; i++) {
            data.nonzeros.emplace_back(i * 3, i * 3, 0.0);
        }
        data.sum_duplicates();
        mtx = Mtx::create(ref);
        mtx->read(data);
        dmtx = gko::clone(exec, mtx);
    }

    std::default_random_engine rng;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> dmtx;
};


TEST_F(SparsityCsr, KernelDiagonalElementPrefixSumIsEquivalentToRef)
{
    gko::array<index_type> prefix_sum{this->ref, this->mtx->get_size()[0] + 1};
    gko::array<index_type> dprefix_sum{this->exec,
                                       this->mtx->get_size()[0] + 1};

    gko::kernels::reference::sparsity_csr::diagonal_element_prefix_sum(
        ref, mtx.get(), prefix_sum.get_data());
    gko::kernels::EXEC_NAMESPACE::sparsity_csr::diagonal_element_prefix_sum(
        exec, dmtx.get(), dprefix_sum.get_data());

    GKO_ASSERT_ARRAY_EQ(prefix_sum, dprefix_sum);
}


TEST_F(SparsityCsr, KernelRemoveDiagonalElementsIsEquivalentToRef)
{
    const auto num_rows = this->mtx->get_size()[0];
    gko::array<index_type> prefix_sum{this->ref, num_rows + 1};
    gko::kernels::reference::sparsity_csr::diagonal_element_prefix_sum(
        ref, mtx.get(), prefix_sum.get_data());
    gko::array<index_type> dprefix_sum{this->exec, prefix_sum};
    const auto out_mtx = Mtx::create(
        ref, mtx->get_size(),
        mtx->get_num_nonzeros() - prefix_sum.get_const_data()[num_rows]);
    const auto dout_mtx = Mtx::create(
        exec, mtx->get_size(),
        mtx->get_num_nonzeros() - prefix_sum.get_const_data()[num_rows]);

    gko::kernels::reference::sparsity_csr::remove_diagonal_elements(
        ref, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
        prefix_sum.get_const_data(), out_mtx.get());
    gko::kernels::EXEC_NAMESPACE::sparsity_csr::remove_diagonal_elements(
        exec, dmtx->get_const_row_ptrs(), dmtx->get_const_col_idxs(),
        dprefix_sum.get_const_data(), dout_mtx.get());

    GKO_ASSERT_MTX_NEAR(out_mtx, dout_mtx, 0.0);
}


TEST_F(SparsityCsr, ToAdjacencyMatrixIsEquivalentToRef)
{
    const auto out_mtx = mtx->to_adjacency_matrix();
    const auto dout_mtx = dmtx->to_adjacency_matrix();

    GKO_ASSERT_MTX_NEAR(out_mtx, dout_mtx, 0.0);
}


TEST_F(SparsityCsr, ConvertToDenseIsEquivalentToRef)
{
    const auto out_dense = gko::matrix::Dense<value_type>::create(
        exec, mtx->get_size(), mtx->get_size()[1] + 2);
    const auto dout_dense = gko::matrix::Dense<value_type>::create(
        exec, mtx->get_size(), mtx->get_size()[1] + 2);

    mtx->convert_to(out_dense);
    dmtx->convert_to(dout_dense);

    GKO_ASSERT_MTX_NEAR(out_dense, dout_dense, 0.0);
}


TEST_F(SparsityCsr, SortSortedMatrixIsEquivalentToRef)
{
    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    auto cols_view =
        gko::make_array_view(ref, mtx->get_num_nonzeros(), mtx->get_col_idxs());
    auto dcols_view = gko::make_array_view(exec, dmtx->get_num_nonzeros(),
                                           dmtx->get_col_idxs());
    GKO_ASSERT_ARRAY_EQ(cols_view, dcols_view);
}


TEST_F(SparsityCsr, SortSortedMatrix64IsEquivalentToRef)
{
    auto mtx64 = Mtx64::create(ref);
    auto dmtx64 = Mtx64::create(exec);
    gko::matrix_data<value_type, index_type> data;
    gko::matrix_data<value_type, gko::int64> data64;
    mtx->sort_by_column_index();
    mtx->write(data);
    data64.size = data.size;
    for (auto entry : data.nonzeros) {
        data64.nonzeros.emplace_back(entry.row, entry.column, entry.value);
    }
    mtx64->read(data64);
    dmtx64->read(data64);

    mtx64->sort_by_column_index();
    dmtx64->sort_by_column_index();

    auto cols_view = gko::make_array_view(ref, mtx64->get_num_nonzeros(),
                                          mtx64->get_col_idxs());
    auto dcols_view = gko::make_array_view(exec, dmtx64->get_num_nonzeros(),
                                           dmtx64->get_col_idxs());
    GKO_ASSERT_ARRAY_EQ(cols_view, dcols_view);
}


TEST_F(SparsityCsr, SortUnsortedMatrixIsEquivalentToRef)
{
    gko::test::unsort_matrix(mtx, rng);
    dmtx->copy_from(mtx);

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    auto cols_view =
        gko::make_array_view(ref, mtx->get_num_nonzeros(), mtx->get_col_idxs());
    auto dcols_view = gko::make_array_view(exec, dmtx->get_num_nonzeros(),
                                           dmtx->get_col_idxs());
    GKO_ASSERT_ARRAY_EQ(cols_view, dcols_view);
}


TEST_F(SparsityCsr, SortUnsortedMatrix64IsEquivalentToRef)
{
    gko::test::unsort_matrix(mtx, rng);
    auto mtx64 = Mtx64::create(ref);
    auto dmtx64 = Mtx64::create(exec);
    gko::matrix_data<value_type, index_type> data;
    gko::matrix_data<value_type, gko::int64> data64;
    mtx->write(data);
    data64.size = data.size;
    for (auto entry : data.nonzeros) {
        data64.nonzeros.emplace_back(entry.row, entry.column, entry.value);
    }
    mtx64->read(data64);
    dmtx64->read(data64);

    mtx64->sort_by_column_index();
    dmtx64->sort_by_column_index();

    auto cols_view = gko::make_array_view(ref, mtx64->get_num_nonzeros(),
                                          mtx64->get_col_idxs());
    auto dcols_view = gko::make_array_view(exec, dmtx64->get_num_nonzeros(),
                                           dmtx64->get_col_idxs());
    GKO_ASSERT_ARRAY_EQ(cols_view, dcols_view);
}


TEST_F(SparsityCsr, RecognizesUnsortedMatrix)
{
    gko::test::unsort_matrix(dmtx, rng);

    ASSERT_FALSE(dmtx->is_sorted_by_column_index());
}


TEST_F(SparsityCsr, RecognizesSortedMatrix)
{
    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
}


}  // namespace

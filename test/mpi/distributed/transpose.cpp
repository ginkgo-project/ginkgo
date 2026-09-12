// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#ifndef GKO_COMPILING_DPCPP


template <typename ValueLocalGlobalIndexType>
class DistTranspose : public CommonMpiTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using dist_mtx =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec = gko::experimental::distributed::Vector<value_type>;
    using local_csr = gko::matrix::Csr<value_type, local_index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;

    // 9x6 on 3 ranks: 3 rows and 2 columns each.
    static constexpr gko::size_type num_rows = 9;
    static constexpr gko::size_type num_cols = 6;

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    // Small values, exact in every tested value type.
    static value_type entry_value(gko::size_type row, gko::size_type col)
    {
        return static_cast<value_type>(static_cast<double>(10 * row + col + 1));
    }

    void add_entry(matrix_data& data, gko::size_type row, gko::size_type col)
    {
        data.nonzeros.emplace_back(static_cast<global_index_type>(row),
                                   static_cast<global_index_type>(col),
                                   entry_value(row, col));
    }

    // Two entries per row, spread so each rank's rows reach every rank's
    // columns, which gives A a real halo.
    matrix_data build_rectangular()
    {
        matrix_data data{gko::dim<2>{num_rows, num_cols}};
        for (gko::size_type row = 0; row < num_rows; ++row) {
            const auto first = row % num_cols;
            const auto second = (2 * row + 1) % num_cols;
            add_entry(data, row, first);
            if (second != first) {
                add_entry(data, row, second);
            }
        }
        data.sort_row_major();
        return data;
    }

    // Rank 1's columns are touched only by rows of ranks 0 and 2, so after
    // transposing its rows come purely from remote contributions and its
    // diagonal block is empty.
    matrix_data build_remote_only()
    {
        matrix_data data{gko::dim<2>{num_rows, num_cols}};
        for (gko::size_type row : {0u, 1u, 2u, 6u, 7u, 8u}) {
            add_entry(data, row, 2);
            add_entry(data, row, 3);
        }
        for (gko::size_type row : {3u, 4u, 5u}) {
            add_entry(data, row, 0);
            add_entry(data, row, 5);
        }
        data.sort_row_major();
        return data;
    }

    matrix_data transpose_data(const matrix_data& data)
    {
        matrix_data transposed{gko::dim<2>{data.size[1], data.size[0]}};
        for (const auto& entry : data.nonzeros) {
            transposed.nonzeros.emplace_back(entry.column, entry.row,
                                             entry.value);
        }
        transposed.sort_row_major();
        return transposed;
    }

    matrix_data build_vector_data(gko::size_type size, unsigned seed)
    {
        matrix_data data{gko::dim<2>{size, 1}};
        for (gko::size_type i = 0; i < size; ++i) {
            data.nonzeros.emplace_back(
                static_cast<global_index_type>(i), 0,
                static_cast<value_type>(static_cast<double>((i * seed) % 7) +
                                        1.0));
        }
        return data;
    }

    void assert_blocks_near(const dist_mtx* actual, const dist_mtx* expected)
    {
        GKO_ASSERT_EQUAL_DIMENSIONS(actual, expected);
        GKO_ASSERT_MTX_NEAR(gko::as<local_csr>(actual->get_diag_matrix()),
                            gko::as<local_csr>(expected->get_diag_matrix()),
                            r<value_type>::value);
        GKO_ASSERT_MTX_NEAR(gko::as<local_csr>(actual->get_off_diag_matrix()),
                            gko::as<local_csr>(expected->get_off_diag_matrix()),
                            r<value_type>::value);
    }
};

TYPED_TEST_SUITE(DistTranspose, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(DistTranspose, RectangularTransposeMatchesSerial)
{
    using dist_mtx = typename TestFixture::dist_mtx;
    using Partition = typename TestFixture::Partition;

    auto nprocs = this->comm.size();
    auto row_part = gko::share(Partition::build_from_global_size_uniform(
        this->exec, nprocs, TestFixture::num_rows));
    auto col_part = gko::share(Partition::build_from_global_size_uniform(
        this->exec, nprocs, TestFixture::num_cols));
    auto a_data = this->build_rectangular();
    auto a = dist_mtx::create(this->exec, this->comm);
    a->read_distributed(a_data, row_part, col_part);
    auto expected = dist_mtx::create(this->exec, this->comm);
    expected->read_distributed(this->transpose_data(a_data), col_part,
                               row_part);
    auto transposed = dist_mtx::create(this->exec, this->comm);

    a->transpose(transposed);

    this->assert_blocks_near(transposed.get(), expected.get());
}


TYPED_TEST(DistTranspose, TransposeWithOnlyRemoteContributionsMatchesSerial)
{
    using dist_mtx = typename TestFixture::dist_mtx;
    using local_csr = typename TestFixture::local_csr;
    using Partition = typename TestFixture::Partition;

    auto nprocs = this->comm.size();
    auto row_part = gko::share(Partition::build_from_global_size_uniform(
        this->exec, nprocs, TestFixture::num_rows));
    auto col_part = gko::share(Partition::build_from_global_size_uniform(
        this->exec, nprocs, TestFixture::num_cols));
    auto a_data = this->build_remote_only();
    auto a = dist_mtx::create(this->exec, this->comm);
    a->read_distributed(a_data, row_part, col_part);
    auto expected = dist_mtx::create(this->exec, this->comm);
    expected->read_distributed(this->transpose_data(a_data), col_part,
                               row_part);
    auto transposed = dist_mtx::create(this->exec, this->comm);

    a->transpose(transposed);

    this->assert_blocks_near(transposed.get(), expected.get());
    // Rank 1 owns the rows that only remote ranks contribute to.
    if (this->comm.rank() == 1) {
        ASSERT_EQ(gko::as<local_csr>(transposed->get_diag_matrix())
                      ->get_num_stored_elements(),
                  gko::size_type{0});
        ASSERT_GT(gko::as<local_csr>(transposed->get_off_diag_matrix())
                      ->get_num_stored_elements(),
                  gko::size_type{0});
    }
}


TYPED_TEST(DistTranspose, TransposeSatisfiesInnerProductIdentity)
{
    using value_type = typename TestFixture::value_type;
    using dist_mtx = typename TestFixture::dist_mtx;
    using dist_vec = typename TestFixture::dist_vec;
    using Dense = typename TestFixture::Dense;
    using Partition = typename TestFixture::Partition;
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);

    auto nprocs = this->comm.size();
    auto row_part = gko::share(Partition::build_from_global_size_uniform(
        this->exec, nprocs, TestFixture::num_rows));
    auto col_part = gko::share(Partition::build_from_global_size_uniform(
        this->exec, nprocs, TestFixture::num_cols));

    auto a = dist_mtx::create(this->exec, this->comm);
    a->read_distributed(this->build_rectangular(), row_part, col_part);
    auto transposed = dist_mtx::create(this->exec, this->comm);
    // x lives on the columns of A, y on its rows.
    auto x = dist_vec::create(this->exec, this->comm);
    x->read_distributed(this->build_vector_data(TestFixture::num_cols, 3),
                        col_part);
    auto y = dist_vec::create(this->exec, this->comm);
    y->read_distributed(this->build_vector_data(TestFixture::num_rows, 5),
                        row_part);
    auto ax = dist_vec::create(this->exec, this->comm);
    ax->read_distributed(this->build_vector_data(TestFixture::num_rows, 0),
                         row_part);
    auto aty = dist_vec::create(this->exec, this->comm);
    aty->read_distributed(this->build_vector_data(TestFixture::num_cols, 0),
                          col_part);
    auto left = Dense::create(this->exec, gko::dim<2>{1, 1});
    auto right = Dense::create(this->exec, gko::dim<2>{1, 1});

    a->transpose(transposed);

    // (A x, y) == (x, A^T y). compute_dot does not conjugate, so this holds
    // for complex values too.
    a->apply(x, ax);
    transposed->apply(y, aty);
    ax->compute_dot(y, left);
    x->compute_dot(aty, right);
    GKO_ASSERT_MTX_NEAR(left, right, r<value_type>::value * 10);
}


TYPED_TEST(DistTranspose, TransposedMatrixIsValidMultiplyOperand)
{
    using value_type = typename TestFixture::value_type;
    using dist_mtx = typename TestFixture::dist_mtx;
    using Partition = typename TestFixture::Partition;
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)
    // multiply runs a local csr::spgemm, which needs 32-bit local indices on
    // CUDA and HIP (see spgemm.cpp).
    if (sizeof(typename TestFixture::local_index_type) > 4) {
        GTEST_SKIP() << "no 64-bit local index spgemm on this backend";
    }
#endif

    auto nprocs = this->comm.size();
    auto row_part = gko::share(Partition::build_from_global_size_uniform(
        this->exec, nprocs, TestFixture::num_rows));
    auto col_part = gko::share(Partition::build_from_global_size_uniform(
        this->exec, nprocs, TestFixture::num_cols));
    auto a_data = this->build_rectangular();
    auto a = dist_mtx::create(this->exec, this->comm);
    a->read_distributed(a_data, row_part, col_part);
    auto transposed = dist_mtx::create(this->exec, this->comm);
    a->transpose(transposed);
    // Comparing against the same product from the serially transposed matrix
    // keeps this value-sensitive; (A^T A) x against A^T (A x) would not,
    // since both sides use the same transpose.
    auto expected_transposed = dist_mtx::create(this->exec, this->comm);
    expected_transposed->read_distributed(this->transpose_data(a_data),
                                          col_part, row_part);
    auto expected_product = dist_mtx::create(this->exec, this->comm);
    expected_transposed->multiply(a, expected_product);
    auto product = dist_mtx::create(this->exec, this->comm);

    // A^T A: the transpose's column partition has to match A's row
    // partition.
    transposed->multiply(a, product);

    ASSERT_EQ(product->get_size(),
              gko::dim<2>(TestFixture::num_cols, TestFixture::num_cols));
    this->assert_blocks_near(product.get(), expected_product.get());
}


TYPED_TEST(DistTranspose, ThrowsIfMatrixHasNoRowPartition)
{
    using dist_mtx = typename TestFixture::dist_mtx;

    auto a = dist_mtx::create(this->exec, this->comm);
    auto transposed = dist_mtx::create(this->exec, this->comm);

    ASSERT_THROW(a->transpose(transposed), gko::InvalidStateError);
}


#endif  // GKO_COMPILING_DPCPP

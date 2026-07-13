// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <random>

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
#include "core/test/utils/matrix_generator.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#ifndef GKO_COMPILING_DPCPP


#ifdef GKO_COMPILING_HIP
#define SKIP_IF_HIP_NO_INT64_SPGEMM(local_index_type)                        \
    if (sizeof(local_index_type) > 4) {                                      \
        GTEST_SKIP() << "distributed spgemm with 64-bit local indices is "   \
                        "unsupported on HIP (rocSPARSE has no 64-bit "       \
                        "spgemm)";                                           \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#else
#define SKIP_IF_HIP_NO_INT64_SPGEMM(local_index_type)                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif


template <typename ValueLocalGlobalIndexType>
class DistSpgemm : public CommonMpiTestFixture {
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
    using global_csr = gko::matrix::Csr<value_type, global_index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    // Build a tridiagonal matrix of the given size with diag=diag_val,
    // off-diag=off_val
    matrix_data build_tridiag(gko::size_type n, value_type diag_val,
                              value_type off_val)
    {
        matrix_data data{gko::dim<2>{n, n}};
        for (gko::size_type i = 0; i < n; ++i) {
            data.nonzeros.emplace_back(i, i, diag_val);
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, off_val);
            }
            if (i + 1 < n) {
                data.nonzeros.emplace_back(i, i + 1, off_val);
            }
        }
        data.sort_row_major();
        return data;
    }

    // Build identity matrix of the given size
    matrix_data build_identity(gko::size_type n)
    {
        matrix_data data{gko::dim<2>{n, n}};
        for (gko::size_type i = 0; i < n; ++i) {
            data.nonzeros.emplace_back(i, i, gko::one<value_type>());
        }
        data.sort_row_major();
        return data;
    }

    // Compute sequential reference: C = A * B using Dense matrices
    // Returns a Dense matrix on ref executor
    std::unique_ptr<Dense> compute_sequential_product(const matrix_data& a_data,
                                                      const matrix_data& b_data)
    {
        auto a_dense = Dense::create(ref);
        a_dense->read(a_data);
        auto b_dense = Dense::create(ref);
        b_dense->read(b_data);
        auto c_dense =
            Dense::create(ref, gko::dim<2>{a_data.size[0], b_data.size[1]});
        a_dense->apply(b_dense, c_dense);
        return c_dense;
    }
};

TYPED_TEST_SUITE(DistSpgemm, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(DistSpgemm, IdentityTimesMatrixIsMatrix)
{
    using value_type = typename TestFixture::value_type;
    using dist_mtx = typename TestFixture::dist_mtx;
    using dist_vec = typename TestFixture::dist_vec;
    using Dense = typename TestFixture::Dense;
    using Partition = typename TestFixture::Partition;
    using global_index_type = typename TestFixture::global_index_type;
    // Half-precision types lack the range for reliable SpGEMM
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
    using local_index_type = typename TestFixture::local_index_type;
    SKIP_IF_HIP_NO_INT64_SPGEMM(local_index_type);

    const gko::size_type n = 6;
    auto nprocs = this->comm.size();
    auto partition = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, n));
    auto identity_data = this->build_identity(n);
    auto tridiag_data =
        this->build_tridiag(n, value_type{2.0}, value_type{-1.0});

    auto identity = dist_mtx::create(this->exec, this->comm);
    identity->read_distributed(identity_data, partition);
    auto a_mat = dist_mtx::create(this->exec, this->comm);
    a_mat->read_distributed(tridiag_data, partition);

    auto c_mat = dist_mtx::create(this->exec, this->comm);
    identity->multiply(a_mat, c_mat);

    // Verify via SpMV: C*x should equal A*x for a random x
    auto x_data =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{n, 1}};
    for (gko::size_type i = 0; i < n; ++i) {
        x_data.nonzeros.emplace_back(
            i, 0, static_cast<value_type>(static_cast<double>(i + 1)));
    }

    auto x_dist = dist_vec::create(this->ref, this->comm);
    x_dist->read_distributed(x_data, partition);
    auto y_c = dist_vec::create(this->ref, this->comm);
    y_c->read_distributed(x_data, partition);
    auto y_a = dist_vec::create(this->ref, this->comm);
    y_a->read_distributed(x_data, partition);

    c_mat->apply(x_dist, y_c);
    a_mat->apply(x_dist, y_a);

    GKO_ASSERT_MTX_NEAR(y_c->get_local_vector(), y_a->get_local_vector(),
                        r<value_type>::value);
}


TYPED_TEST(DistSpgemm, MatrixTimesIdentityIsMatrix)
{
    using value_type = typename TestFixture::value_type;
    using dist_mtx = typename TestFixture::dist_mtx;
    using dist_vec = typename TestFixture::dist_vec;
    using Dense = typename TestFixture::Dense;
    using Partition = typename TestFixture::Partition;
    using global_index_type = typename TestFixture::global_index_type;
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
    using local_index_type = typename TestFixture::local_index_type;
    SKIP_IF_HIP_NO_INT64_SPGEMM(local_index_type);

    const gko::size_type n = 6;
    auto nprocs = this->comm.size();
    auto partition = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, n));
    auto identity_data = this->build_identity(n);
    auto tridiag_data =
        this->build_tridiag(n, value_type{2.0}, value_type{-1.0});

    auto identity = dist_mtx::create(this->exec, this->comm);
    identity->read_distributed(identity_data, partition);
    auto a_mat = dist_mtx::create(this->exec, this->comm);
    a_mat->read_distributed(tridiag_data, partition);

    auto c_mat = dist_mtx::create(this->exec, this->comm);
    a_mat->multiply(identity, c_mat);

    // Verify via SpMV: C*x should equal A*x for a random x
    auto x_data =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{n, 1}};
    for (gko::size_type i = 0; i < n; ++i) {
        x_data.nonzeros.emplace_back(
            i, 0, static_cast<value_type>(static_cast<double>(i + 1)));
    }

    auto x_dist = dist_vec::create(this->ref, this->comm);
    x_dist->read_distributed(x_data, partition);
    auto y_c = dist_vec::create(this->ref, this->comm);
    y_c->read_distributed(x_data, partition);
    auto y_a = dist_vec::create(this->ref, this->comm);
    y_a->read_distributed(x_data, partition);

    c_mat->apply(x_dist, y_c);
    a_mat->apply(x_dist, y_a);

    GKO_ASSERT_MTX_NEAR(y_c->get_local_vector(), y_a->get_local_vector(),
                        r<value_type>::value);
}


TYPED_TEST(DistSpgemm, RandomSparseMatchesSequential)
{
    using value_type = typename TestFixture::value_type;
    using dist_mtx = typename TestFixture::dist_mtx;
    using dist_vec = typename TestFixture::dist_vec;
    using Dense = typename TestFixture::Dense;
    using Partition = typename TestFixture::Partition;
    using global_csr = typename TestFixture::global_csr;
    using global_index_type = typename TestFixture::global_index_type;
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
    using local_index_type = typename TestFixture::local_index_type;
    SKIP_IF_HIP_NO_INT64_SPGEMM(local_index_type);

    const gko::size_type n = 12;
    auto nprocs = this->comm.size();
    auto partition = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, n));

    // All ranks generate the same matrices (same seed)
    auto a_data =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            n, n, std::uniform_int_distribution<>(1, 4),
            std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(42));
    auto b_data =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            n, n, std::uniform_int_distribution<>(1, 4),
            std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(123));

    // Sequential reference: C_seq = A_seq * B_seq using Dense
    auto c_seq = this->compute_sequential_product(a_data, b_data);

    // Distributed: partition, read_distributed, spgemm
    auto a_dist = dist_mtx::create(this->exec, this->comm);
    a_dist->read_distributed(a_data, partition);
    auto b_dist = dist_mtx::create(this->exec, this->comm);
    b_dist->read_distributed(b_data, partition);

    auto c_mat = dist_mtx::create(this->exec, this->comm);
    a_dist->multiply(b_dist, c_mat);

    // Verify via SpMV comparison
    auto x_data =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{n, 1}};
    for (gko::size_type i = 0; i < n; ++i) {
        x_data.nonzeros.emplace_back(
            i, 0, static_cast<value_type>(static_cast<double>(i + 1)));
    }

    // Sequential SpMV: y_ref = C_seq * x
    auto x_full = Dense::create(this->ref);
    x_full->read(x_data);
    auto y_ref = Dense::create(this->ref, gko::dim<2>{n, 1});
    c_seq->apply(x_full, y_ref);

    // Distributed SpMV: y_dist = C_dist * x_dist
    auto x_dist = dist_vec::create(this->ref, this->comm);
    x_dist->read_distributed(x_data, partition);
    auto y_dist = dist_vec::create(this->ref, this->comm);
    y_dist->read_distributed(x_data, partition);
    c_mat->apply(x_dist, y_dist);

    // Gather and compare: each rank checks its local portion
    auto rank = this->comm.rank();
    auto host_part = gko::clone(this->ref, partition);
    auto range_bounds = host_part->get_range_bounds();
    auto part_ids = host_part->get_part_ids();
    std::vector<global_index_type> gather_idxs;
    for (gko::size_type range_id = 0; range_id < host_part->get_num_ranges();
         ++range_id) {
        if (part_ids[range_id] == rank) {
            for (global_index_type row = range_bounds[range_id];
                 row < range_bounds[range_id + 1]; ++row) {
                gather_idxs.push_back(row);
            }
        }
    }
    gko::array<global_index_type> gather_arr(this->ref, gather_idxs.begin(),
                                             gather_idxs.end());
    auto y_ref_local = y_ref->row_gather(&gather_arr);

    GKO_ASSERT_MTX_NEAR(y_dist->get_local_vector(), y_ref_local,
                        r<value_type>::value * 10);
}


TYPED_TEST(DistSpgemm, NonSquareMismatchedPartitions)
{
    using value_type = typename TestFixture::value_type;
    using dist_mtx = typename TestFixture::dist_mtx;
    using dist_vec = typename TestFixture::dist_vec;
    using Partition = typename TestFixture::Partition;
    using global_index_type = typename TestFixture::global_index_type;
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
    using local_index_type = typename TestFixture::local_index_type;
    SKIP_IF_HIP_NO_INT64_SPGEMM(local_index_type);

    const gko::size_type m = 6;   // A rows
    const gko::size_type k = 9;   // A cols = B rows
    const gko::size_type n = 12;  // B cols
    auto nprocs = this->comm.size();

    // A is m x k: row partition on m, col partition on k
    // B is k x n: row partition on k (= A col partition), col partition on n
    // Ensures A.col_partition == B.row_partition and all three partitions
    // differ in size, exercising the mismatched-partition code paths.
    auto a_row_part = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, m));
    auto a_col_part = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, k));
    auto b_row_part = a_col_part;
    auto b_col_part = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, n));

    auto a_data =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            m, k, std::uniform_int_distribution<>(1, 3),
            std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(42));
    auto b_data =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            k, n, std::uniform_int_distribution<>(1, 3),
            std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(123));

    auto a_dist = dist_mtx::create(this->exec, this->comm);
    a_dist->read_distributed(a_data, a_row_part, a_col_part);
    auto b_dist = dist_mtx::create(this->exec, this->comm);
    b_dist->read_distributed(b_data, b_row_part, b_col_part);

    auto c_mat = dist_mtx::create(this->exec, this->comm);
    a_dist->multiply(b_dist, c_mat);

    // Result dimensions: m x n
    ASSERT_EQ(c_mat->get_size()[0], m);
    ASSERT_EQ(c_mat->get_size()[1], n);

    // Verify correctness: C*x should equal the sequential product's action
    // on x for a deterministic x.
    auto x_data =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{n, 1}};
    for (gko::size_type i = 0; i < n; ++i) {
        x_data.nonzeros.emplace_back(
            i, 0, static_cast<value_type>(static_cast<double>(i + 1)));
    }

    auto x_dist = dist_vec::create(this->ref, this->comm);
    x_dist->read_distributed(x_data, b_col_part);
    auto y_dist = dist_vec::create(this->ref, this->comm);
    auto y_init =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{m, 1}};
    for (gko::size_type i = 0; i < m; ++i) {
        y_init.nonzeros.emplace_back(i, 0, gko::zero<value_type>());
    }
    y_dist->read_distributed(y_init, a_row_part);
    c_mat->apply(x_dist, y_dist);

    // Reference: compute C_ref = A*B as a sequential Dense, then C_ref*x
    auto c_ref = this->compute_sequential_product(a_data, b_data);
    auto x_ref =
        gko::matrix::Dense<value_type>::create(this->ref, gko::dim<2>{n, 1});
    x_ref->read(x_data);
    auto y_ref =
        gko::matrix::Dense<value_type>::create(this->ref, gko::dim<2>{m, 1});
    c_ref->apply(x_ref, y_ref);

    // Compare only this rank's slice of y
    auto rank = this->comm.rank();
    auto a_row_part_host = gko::clone(this->ref, a_row_part);
    auto local_m_begin = a_row_part_host->get_range_bounds()[rank];
    auto local_m_end = a_row_part_host->get_range_bounds()[rank + 1];
    auto y_ref_slice = y_ref->create_submatrix(
        gko::span{static_cast<gko::size_type>(local_m_begin),
                  static_cast<gko::size_type>(local_m_end)},
        gko::span{0, 1});
    GKO_ASSERT_MTX_NEAR(y_dist->get_local_vector(), y_ref_slice,
                        r<value_type>::value * 100);
}


TYPED_TEST(DistSpgemm, NonContiguousInnerPartitionMatchesSequential)
{
    using value_type = typename TestFixture::value_type;
    using dist_mtx = typename TestFixture::dist_mtx;
    using dist_vec = typename TestFixture::dist_vec;
    using Dense = typename TestFixture::Dense;
    using Partition = typename TestFixture::Partition;
    using global_index_type = typename TestFixture::global_index_type;
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
    using local_index_type = typename TestFixture::local_index_type;
    SKIP_IF_HIP_NO_INT64_SPGEMM(local_index_type);

    const gko::size_type m = 6;  // A rows
    const gko::size_type k = 9;  // A cols = B rows (shared inner dimension)
    const gko::size_type n = 5;  // B cols
    auto nprocs = this->comm.size();

    auto a_row_part = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, m));
    auto b_col_part = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, n));

    // Non-contiguous partition of the shared inner dimension: round-robin
    // assignment means each part's global indices are scattered across
    // several disjoint ranges rather than a single contiguous block. This
    // is used for both A's column partition and B's row partition, since
    // matmul compatibility requires them to match.
    gko::array<gko::experimental::mpi::comm_index_type> inner_mapping{
        this->exec, {0, 1, 2, 0, 1, 2, 0, 1, 2}};
    auto inner_part = gko::share(
        Partition::build_from_mapping(this->exec, inner_mapping, nprocs));

    auto a_data =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            m, k, std::uniform_int_distribution<>(1, 4),
            std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(7));
    auto b_data =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            k, n, std::uniform_int_distribution<>(1, 4),
            std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(11));

    auto a_dist = dist_mtx::create(this->exec, this->comm);
    a_dist->read_distributed(a_data, a_row_part, inner_part);
    auto b_dist = dist_mtx::create(this->exec, this->comm);
    b_dist->read_distributed(b_data, inner_part, b_col_part);

    auto c_mat = dist_mtx::create(this->exec, this->comm);
    a_dist->multiply(b_dist, c_mat);

    ASSERT_EQ(c_mat->get_size()[0], m);
    ASSERT_EQ(c_mat->get_size()[1], n);

    // Sequential reference: C_seq = A * B, then y_ref = C_seq * x
    auto c_seq = this->compute_sequential_product(a_data, b_data);
    auto x_data =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{n, 1}};
    for (gko::size_type i = 0; i < n; ++i) {
        x_data.nonzeros.emplace_back(
            i, 0, static_cast<value_type>(static_cast<double>(i + 1)));
    }
    auto x_full = Dense::create(this->ref);
    x_full->read(x_data);
    auto y_ref = Dense::create(this->ref, gko::dim<2>{m, 1});
    c_seq->apply(x_full, y_ref);

    // Distributed SpMV
    auto x_dist = dist_vec::create(this->ref, this->comm);
    x_dist->read_distributed(x_data, b_col_part);
    auto y_dist = dist_vec::create(this->ref, this->comm);
    auto y_init =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{m, 1}};
    for (gko::size_type i = 0; i < m; ++i) {
        y_init.nonzeros.emplace_back(i, 0, gko::zero<value_type>());
    }
    y_dist->read_distributed(y_init, a_row_part);
    c_mat->apply(x_dist, y_dist);

    // Gather this rank's rows out of the sequential reference and compare.
    // Uses the general range-scan gather (not a rank-indexed range lookup)
    // since a_row_part need not have exactly one range per rank.
    auto rank = this->comm.rank();
    auto host_part = gko::clone(this->ref, a_row_part);
    auto range_bounds = host_part->get_range_bounds();
    auto part_ids = host_part->get_part_ids();
    std::vector<global_index_type> gather_idxs;
    for (gko::size_type range_id = 0; range_id < host_part->get_num_ranges();
         ++range_id) {
        if (part_ids[range_id] == rank) {
            for (global_index_type row = range_bounds[range_id];
                 row < range_bounds[range_id + 1]; ++row) {
                gather_idxs.push_back(row);
            }
        }
    }
    gko::array<global_index_type> gather_arr(this->ref, gather_idxs.begin(),
                                             gather_idxs.end());
    auto y_ref_local = y_ref->row_gather(&gather_arr);

    GKO_ASSERT_MTX_NEAR(y_dist->get_local_vector(), y_ref_local,
                        r<value_type>::value * 10);
}


TYPED_TEST(DistSpgemm, EmptyLocalRowsMatchesSequential)
{
    using value_type = typename TestFixture::value_type;
    using dist_mtx = typename TestFixture::dist_mtx;
    using dist_vec = typename TestFixture::dist_vec;
    using Dense = typename TestFixture::Dense;
    using Partition = typename TestFixture::Partition;
    using global_index_type = typename TestFixture::global_index_type;
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
    using local_index_type = typename TestFixture::local_index_type;
    SKIP_IF_HIP_NO_INT64_SPGEMM(local_index_type);

    const gko::size_type m = 6;  // A rows (= C rows)
    const gko::size_type k = 8;  // A cols = B rows (shared inner dimension)
    const gko::size_type n = 5;  // B cols
    auto nprocs = this->comm.size();
    ASSERT_EQ(nprocs, 3);

    // A's row partition deliberately leaves rank 1 with zero local rows:
    // all of A/C's rows are split only between ranks 0 and 2.
    gko::array<gko::experimental::mpi::comm_index_type> row_mapping{
        this->exec, {0, 0, 0, 2, 2, 2}};
    auto a_row_part = gko::share(
        Partition::build_from_mapping(this->exec, row_mapping, nprocs));

    auto inner_part = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, k));
    auto b_col_part = gko::share(
        Partition::build_from_global_size_uniform(this->exec, nprocs, n));

    auto a_data =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            m, k, std::uniform_int_distribution<>(1, 4),
            std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(13));
    auto b_data =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            k, n, std::uniform_int_distribution<>(1, 4),
            std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(17));

    auto a_dist = dist_mtx::create(this->exec, this->comm);
    a_dist->read_distributed(a_data, a_row_part, inner_part);
    auto b_dist = dist_mtx::create(this->exec, this->comm);
    b_dist->read_distributed(b_data, inner_part, b_col_part);

    // Sanity check: rank 1 indeed owns zero local rows of A (and thus of C).
    if (this->comm.rank() == 1) {
        ASSERT_EQ(a_dist->get_diag_matrix()->get_size()[0], 0u);
    }

    auto c_mat = dist_mtx::create(this->exec, this->comm);
    a_dist->multiply(b_dist, c_mat);

    ASSERT_EQ(c_mat->get_size()[0], m);
    ASSERT_EQ(c_mat->get_size()[1], n);

    // Sequential reference: C_seq = A * B, then y_ref = C_seq * x
    auto c_seq = this->compute_sequential_product(a_data, b_data);
    auto x_data =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{n, 1}};
    for (gko::size_type i = 0; i < n; ++i) {
        x_data.nonzeros.emplace_back(
            i, 0, static_cast<value_type>(static_cast<double>(i + 1)));
    }
    auto x_full = Dense::create(this->ref);
    x_full->read(x_data);
    auto y_ref = Dense::create(this->ref, gko::dim<2>{m, 1});
    c_seq->apply(x_full, y_ref);

    // Distributed SpMV
    auto x_dist = dist_vec::create(this->ref, this->comm);
    x_dist->read_distributed(x_data, b_col_part);
    auto y_dist = dist_vec::create(this->ref, this->comm);
    auto y_init =
        gko::matrix_data<value_type, global_index_type>{gko::dim<2>{m, 1}};
    for (gko::size_type i = 0; i < m; ++i) {
        y_init.nonzeros.emplace_back(i, 0, gko::zero<value_type>());
    }
    y_dist->read_distributed(y_init, a_row_part);
    c_mat->apply(x_dist, y_dist);

    // Gather this rank's rows out of the sequential reference and compare.
    // For rank 1 this yields an empty selection, exercising the
    // zero-local-rows path end to end.
    auto rank = this->comm.rank();
    auto host_part = gko::clone(this->ref, a_row_part);
    auto range_bounds = host_part->get_range_bounds();
    auto part_ids = host_part->get_part_ids();
    std::vector<global_index_type> gather_idxs;
    for (gko::size_type range_id = 0; range_id < host_part->get_num_ranges();
         ++range_id) {
        if (part_ids[range_id] == rank) {
            for (global_index_type row = range_bounds[range_id];
                 row < range_bounds[range_id + 1]; ++row) {
                gather_idxs.push_back(row);
            }
        }
    }
    gko::array<global_index_type> gather_arr(this->ref, gather_idxs.begin(),
                                             gather_idxs.end());
    auto y_ref_local = y_ref->row_gather(&gather_arr);

    GKO_ASSERT_MTX_NEAR(y_dist->get_local_vector(), y_ref_local,
                        r<value_type>::value * 10);
}


#endif

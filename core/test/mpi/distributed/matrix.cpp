// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>

#include "core/test/utils.hpp"


namespace {


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueType, typename IndexType>
class CustomLinOp
    : public gko::LinOp,
      public gko::EnableCloneable<CustomLinOp<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType>,
      public gko::EnableCreateMethod<CustomLinOp<ValueType, IndexType>> {
public:
    void read(const gko::matrix_data<ValueType, IndexType>& data) override {}

    explicit CustomLinOp(std::shared_ptr<const gko::Executor> exec)
        : gko::LinOp(exec)
    {}

    explicit CustomLinOp(std::shared_ptr<const gko::Executor> exec,
                         gko::dim<2> size)
        : gko::LinOp(exec, size)
    {}

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};


template <typename ValueLocalGlobalIndexType>
class MatrixBuilder : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;

    MatrixBuilder()
        : ref(gko::ReferenceExecutor::create()),
          comm(gko::experimental::mpi::communicator(MPI_COMM_WORLD))
    {}

    void SetUp() override { ASSERT_EQ(this->comm.size(), 3); }

    template <typename F>
    void forall_matrix_types(F&& f)
    {
        using namespace gko::matrix;
        auto empty_test = [](gko::ptr_param<const gko::LinOp>) {};
        {
            SCOPED_TRACE("With Coo");
            f(gko::with_matrix_type<Coo>(),
              Coo<value_type, local_index_type>::create(this->ref), empty_test);
        }
        {
            SCOPED_TRACE("With Csr");
            f(gko::with_matrix_type<Csr>(),
              Csr<value_type, local_index_type>::create(this->ref), empty_test);
        }
        {
            SCOPED_TRACE("With Csr with strategy");
            using ConcreteCsr = Csr<value_type, local_index_type>;
            auto strategy = gko::matrix::csr::spmv_strategy::classical;
            f(gko::with_matrix_type<Csr>(strategy),
              ConcreteCsr::create(this->ref, strategy),
              [](gko::ptr_param<const gko::LinOp> local_mat) {
                  auto local_csr = gko::as<ConcreteCsr>(local_mat);

                  ASSERT_EQ(local_csr->get_strategy(),
                            gko::matrix::csr::spmv_strategy::classical);
              });
        }
        {
            SCOPED_TRACE("With Ell");
            f(gko::with_matrix_type<Ell>(),
              Ell<value_type, local_index_type>::create(this->ref), empty_test);
        }
        {
            SCOPED_TRACE("With Fbcsr");
            f(gko::with_matrix_type<Fbcsr>(),
              Fbcsr<value_type, local_index_type>::create(this->ref),
              empty_test);
        }
        {
            SCOPED_TRACE("With Fbcsr with block_size");
            f(gko::with_matrix_type<Fbcsr>(5),
              Fbcsr<value_type, local_index_type>::create(this->ref, 5),
              [](gko::ptr_param<const gko::LinOp> local_mat) {
                  auto local_fbcsr =
                      gko::as<Fbcsr<value_type, local_index_type>>(local_mat);

                  ASSERT_EQ(local_fbcsr->get_block_size(), 5);
              });
        }
        {
            SCOPED_TRACE("With Hybrid");
            f(gko::with_matrix_type<Hybrid>(),
              Hybrid<value_type, local_index_type>::create(this->ref),
              empty_test);
        }
        {
            SCOPED_TRACE("With Hybrid with strategy");
            using Concrete = Hybrid<value_type, local_index_type>;
            auto strategy =
                std::make_shared<typename Concrete::column_limit>(11);
            f(gko::with_matrix_type<Hybrid>(strategy),
              Concrete::create(this->ref, strategy),
              [](gko::ptr_param<const gko::LinOp> local_mat) {
                  auto local_hy = gko::as<Concrete>(local_mat);

                  ASSERT_NO_THROW(gko::as<typename Concrete::column_limit>(
                      local_hy->get_strategy()));
                  ASSERT_EQ(gko::as<typename Concrete::column_limit>(
                                local_hy->get_strategy())
                                ->get_num_columns(),
                            11);
              });
        }
        {
            SCOPED_TRACE("With Sellp");
            f(gko::with_matrix_type<Sellp>(),
              Sellp<value_type, local_index_type>::create(this->ref),
              empty_test);
        }
    }

    template <typename DiagMatrixType, typename OffDiagMatrixType>
    void expected_interface_no_throw(gko::ptr_param<dist_mtx_type> mat,
                                     DiagMatrixType&& diag_matrix_type,
                                     OffDiagMatrixType&& off_diag_matrix_type)
    {
        auto num_rows = mat->get_size()[0];
        auto a = dist_vec_type::create(ref, comm);
        auto b = dist_vec_type::create(ref, comm);
        auto convert_result = dist_mtx_type::create(ref, comm, diag_matrix_type,
                                                    off_diag_matrix_type);
        auto move_result = dist_mtx_type::create(ref, comm, diag_matrix_type,
                                                 off_diag_matrix_type);

        ASSERT_NO_THROW(mat->apply(a, b));
        ASSERT_NO_THROW(mat->convert_to(convert_result));
        ASSERT_NO_THROW(mat->move_to(move_result));
    }


    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::experimental::mpi::communicator comm;
};

TYPED_TEST_SUITE(MatrixBuilder, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(MatrixBuilder, BuildWithLocal)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::local_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    this->forall_matrix_types([this](auto with_matrix_type,
                                     auto expected_type_ptr,
                                     auto additional_test) {
        using expected_type = typename std::remove_pointer<
            decltype(expected_type_ptr.get())>::type;

        auto mat =
            dist_mtx_type::create(this->ref, this->comm, with_matrix_type);

        ASSERT_NO_THROW(gko::as<expected_type>(mat->get_diag_matrix()));
        additional_test(mat->get_diag_matrix());
        additional_test(mat->get_off_diag_matrix());
        this->expected_interface_no_throw(mat, with_matrix_type,
                                          with_matrix_type);
    });
}


TYPED_TEST(MatrixBuilder, BuildWithDiagAndOffDiag)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::local_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    this->forall_matrix_types([this](auto with_diag_matrix_type,
                                     auto expected_diag_type_ptr,
                                     auto additional_diag_test) {
        using expected_diag_type = typename std::remove_pointer<
            decltype(expected_diag_type_ptr.get())>::type;
        this->forall_matrix_types([&](auto with_off_diag_matrix_type,
                                      auto expected_off_diag_type_ptr,
                                      auto additional_off_diag_test) {
            using expected_off_diag_type = typename std::remove_pointer<
                decltype(expected_off_diag_type_ptr.get())>::type;

            auto mat = dist_mtx_type::create(this->ref, this->comm,
                                             with_diag_matrix_type,
                                             with_off_diag_matrix_type);

            ASSERT_NO_THROW(
                gko::as<expected_diag_type>(mat->get_diag_matrix()));
            ASSERT_NO_THROW(
                gko::as<expected_off_diag_type>(mat->get_off_diag_matrix()));
            additional_diag_test(mat->get_diag_matrix());
            additional_off_diag_test(mat->get_off_diag_matrix());
            this->expected_interface_no_throw(mat, with_diag_matrix_type,
                                              with_off_diag_matrix_type);
        });
    });
}


TYPED_TEST(MatrixBuilder, BuildWithCustomLinOp)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::local_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using custom_type = CustomLinOp<value_type, index_type>;

    auto mat = dist_mtx_type::create(this->ref, this->comm,
                                     gko::with_matrix_type<CustomLinOp>());

    ASSERT_NO_THROW(gko::as<custom_type>(mat->get_diag_matrix()));
    this->expected_interface_no_throw(mat, gko::with_matrix_type<CustomLinOp>(),
                                      gko::with_matrix_type<CustomLinOp>());
}


TYPED_TEST(MatrixBuilder, BuildLocalOnly)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::local_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using custom_type = CustomLinOp<value_type, index_type>;
    using empty_off_diag_type = gko::matrix::Coo<value_type, index_type>;
    auto local_n = this->comm.rank() + 1;
    // global_size = 1 + 2 + ... + num_rank
    auto global_n = ((1 + this->comm.size()) * this->comm.size()) / 2;
    auto a =
        dist_vec_type::create(this->ref, this->comm, gko::dim<2>(global_n, 1),
                              gko::dim<2>(local_n, 1));
    auto b =
        dist_vec_type::create(this->ref, this->comm, gko::dim<2>(global_n, 1),
                              gko::dim<2>(local_n, 1));

    auto mat = dist_mtx_type::create(
        this->ref, this->comm, gko::dim<2>(global_n, global_n),
        custom_type::create(this->ref, gko::dim<2>(local_n, local_n)));

    ASSERT_NO_THROW(gko::as<custom_type>(mat->get_diag_matrix()));
    ASSERT_NE(mat->get_off_diag_matrix(), nullptr);
    ASSERT_NO_THROW(gko::as<empty_off_diag_type>(mat->get_off_diag_matrix()));
    GKO_ASSERT_EQUAL_DIMENSIONS(mat->get_diag_matrix()->get_size(),
                                gko::dim<2>(local_n, local_n));
    ASSERT_NO_THROW(mat->apply(a, b));
}


TYPED_TEST(MatrixBuilder, BuildFromLinOpLocal)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::local_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    this->forall_matrix_types([this](auto with_matrix_type,
                                     auto expected_type_ptr,
                                     auto additional_test) {
        using expected_type = typename std::remove_pointer<
            decltype(expected_type_ptr.get())>::type;

        auto mat =
            dist_mtx_type::create(this->ref, this->comm, expected_type_ptr);

        ASSERT_NO_THROW(gko::as<expected_type>(mat->get_diag_matrix()));
        additional_test(mat->get_diag_matrix());
        additional_test(mat->get_off_diag_matrix());
        this->expected_interface_no_throw(mat, with_matrix_type,
                                          with_matrix_type);
    });
}


TYPED_TEST(MatrixBuilder, BuildFromLinOpDiagAndOffDiag)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::local_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    this->forall_matrix_types([this](auto with_diag_matrix_type,
                                     auto expected_diag_type_ptr,
                                     auto additional_diag_test) {
        using expected_diag_type = typename std::remove_pointer<
            decltype(expected_diag_type_ptr.get())>::type;
        this->forall_matrix_types([&](auto with_off_diag_matrix_type,
                                      auto expected_off_diag_type_ptr,
                                      auto additional_off_diag_test) {
            using expected_off_diag_type = typename std::remove_pointer<
                decltype(expected_off_diag_type_ptr.get())>::type;

            auto mat = dist_mtx_type::create(this->ref, this->comm,
                                             expected_diag_type_ptr,
                                             expected_off_diag_type_ptr);

            ASSERT_NO_THROW(
                gko::as<expected_diag_type>(mat->get_diag_matrix()));
            ASSERT_NO_THROW(
                gko::as<expected_off_diag_type>(mat->get_off_diag_matrix()));
            additional_diag_test(mat->get_diag_matrix());
            additional_off_diag_test(mat->get_off_diag_matrix());
            this->expected_interface_no_throw(mat, with_diag_matrix_type,
                                              with_off_diag_matrix_type);
        });
    });
}


TYPED_TEST(MatrixBuilder, WritesMatrixDataWithGlobalIndices)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using mtx_type = gko::matrix::Csr<value_type, local_index_type>;
    using partition_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;
    using writable_type =
        gko::WritableToMatrixData<value_type, global_index_type>;
    const auto rank = this->comm.rank();
    const auto next_rank = (rank + 1) % this->comm.size();
    const auto global_size =
        static_cast<global_index_type>(2 * this->comm.size());
    auto partition = gko::share(partition_type::build_from_global_size_uniform(
        this->ref, this->comm.size(), global_size));
    auto diag = mtx_type::create(this->ref);
    auto off_diag = mtx_type::create(this->ref);
    const auto global_row = static_cast<global_index_type>(2 * rank);
    const auto remote_col = static_cast<global_index_type>(2 * next_rank);
    diag->read(gko::matrix_data<value_type, local_index_type>{
        gko::dim<2>{2, 2},
        {{0, 0, value_type{1.0f}}, {1, 1, value_type{2.0f}}}});
    off_diag->read(gko::matrix_data<value_type, local_index_type>{
        gko::dim<2>{2, 2},
        {{0, 0, value_type{3.0f}}, {1, 1, value_type{4.0f}}}});
    auto remote_cols = gko::array<global_index_type>{
        this->ref,
        {remote_col, static_cast<global_index_type>(remote_col + 1)}};
    gko::matrix_data<value_type, global_index_type> expected{
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)},
        {{global_row, global_row, value_type{1.0f}},
         {static_cast<global_index_type>(global_row + 1),
          static_cast<global_index_type>(global_row + 1), value_type{2.0f}},
         {global_row, remote_col, value_type{3.0f}},
         {static_cast<global_index_type>(global_row + 1),
          static_cast<global_index_type>(remote_col + 1), value_type{4.0f}}}};
    expected.sort_row_major();

    auto imap = map_type{this->ref, partition, rank, remote_cols};
    auto matrix = dist_mtx_type::create(this->ref, this->comm, std::move(imap),
                                        std::move(diag), std::move(off_diag));
    gko::matrix_data<value_type, global_index_type> written;
    gko::as<writable_type>(matrix.get())->write(written);

    ASSERT_EQ(written.size, expected.size);
    ASSERT_EQ(written.nonzeros.size(), expected.nonzeros.size());
    for (gko::size_type i = 0; i < expected.nonzeros.size(); ++i) {
        EXPECT_EQ(written.nonzeros[i], expected.nonzeros[i]);
    }
}


TYPED_TEST(MatrixBuilder, WritesMatrixDataWithNonUniformPartition)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using mtx_type = gko::matrix::Csr<value_type, local_index_type>;
    using partition_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;
    using writable_type =
        gko::WritableToMatrixData<value_type, global_index_type>;
    const auto rank = this->comm.rank();
    auto mapping = gko::array<comm_index_type>{this->ref, {2, 0, 2, 0, 1, 2}};
    auto partition =
        gko::share(partition_type::build_from_mapping(this->ref, mapping, 3));
    auto local_size = partition->get_part_size(rank);
    auto local_dim = gko::dim<2>{static_cast<gko::size_type>(local_size),
                                 static_cast<gko::size_type>(local_size)};
    auto diag = mtx_type::create(this->ref);
    auto off_diag = mtx_type::create(this->ref);
    gko::matrix_data<value_type, local_index_type> diag_data{local_dim, {}};
    gko::matrix_data<value_type, local_index_type> off_diag_data{
        gko::dim<2>{local_dim[0], 1}, {}};
    gko::matrix_data<value_type, global_index_type> expected{gko::dim<2>{6, 6},
                                                             {}};
    auto remote_cols = gko::array<global_index_type>{this->ref};
    if (rank == 0) {
        diag_data.nonzeros = {{0, 0, value_type{1.0f}},
                              {1, 0, value_type{2.0f}}};
        off_diag_data.nonzeros = {{0, 0, value_type{3.0f}}};
        remote_cols =
            gko::array<global_index_type>{this->ref, {global_index_type{4}}};
        expected.nonzeros = {{1, 1, value_type{1.0f}},
                             {1, 4, value_type{3.0f}},
                             {3, 1, value_type{2.0f}}};
    } else if (rank == 1) {
        diag_data.nonzeros = {{0, 0, value_type{1.0f}}};
        off_diag_data.nonzeros = {{0, 0, value_type{2.0f}}};
        remote_cols =
            gko::array<global_index_type>{this->ref, {global_index_type{0}}};
        expected.nonzeros = {{4, 0, value_type{2.0f}},
                             {4, 4, value_type{1.0f}}};
    } else {
        diag_data.nonzeros = {{0, 0, value_type{1.0f}},
                              {1, 0, value_type{2.0f}},
                              {2, 1, value_type{3.0f}}};
        off_diag_data.nonzeros = {{0, 0, value_type{4.0f}}};
        remote_cols =
            gko::array<global_index_type>{this->ref, {global_index_type{1}}};
        expected.nonzeros = {{0, 0, value_type{1.0f}},
                             {0, 1, value_type{4.0f}},
                             {2, 0, value_type{2.0f}},
                             {5, 2, value_type{3.0f}}};
    }
    diag->read(diag_data);
    off_diag->read(off_diag_data);
    expected.sort_row_major();

    auto imap = map_type{this->ref, partition, rank, remote_cols};
    auto matrix = dist_mtx_type::create(this->ref, this->comm, std::move(imap),
                                        std::move(diag), std::move(off_diag));
    gko::matrix_data<value_type, global_index_type> written;
    gko::as<writable_type>(matrix.get())->write(written);

    ASSERT_EQ(written.size, expected.size);
    ASSERT_EQ(written.nonzeros.size(), expected.nonzeros.size());
    for (gko::size_type i = 0; i < expected.nonzeros.size(); ++i) {
        EXPECT_EQ(written.nonzeros[i], expected.nonzeros[i]);
    }
}


TYPED_TEST(MatrixBuilder, WritesEmptyMatrixDataWithGlobalSize)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using mtx_type = gko::matrix::Csr<value_type, local_index_type>;
    using partition_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;
    using writable_type =
        gko::WritableToMatrixData<value_type, global_index_type>;
    const auto rank = this->comm.rank();
    const auto global_size =
        static_cast<global_index_type>(2 * this->comm.size());
    auto partition = gko::share(partition_type::build_from_global_size_uniform(
        this->ref, this->comm.size(), global_size));
    auto diag = mtx_type::create(this->ref);
    auto off_diag = mtx_type::create(this->ref);
    diag->read(
        gko::matrix_data<value_type, local_index_type>{gko::dim<2>{2, 2}, {}});
    off_diag->read(
        gko::matrix_data<value_type, local_index_type>{gko::dim<2>{2, 0}, {}});
    auto remote_cols = gko::array<global_index_type>{this->ref};
    auto imap = map_type{this->ref, partition, rank, remote_cols};
    auto matrix = dist_mtx_type::create(this->ref, this->comm, std::move(imap),
                                        std::move(diag), std::move(off_diag));
    gko::matrix_data<value_type, global_index_type> written;

    gko::as<writable_type>(matrix.get())->write(written);

    ASSERT_EQ(written.size,
              (gko::dim<2>{static_cast<gko::size_type>(global_size),
                           static_cast<gko::size_type>(global_size)}));
    ASSERT_TRUE(written.nonzeros.empty());
}


TYPED_TEST(MatrixBuilder, WritesMatrixDataFromDiagOnlyConstructor)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using mtx_type = gko::matrix::Csr<value_type, local_index_type>;
    using writable_type =
        gko::WritableToMatrixData<value_type, global_index_type>;
    const auto rank = this->comm.rank();
    const auto local_size = static_cast<local_index_type>(rank + 1);
    const auto global_size = static_cast<global_index_type>(
        ((1 + this->comm.size()) * this->comm.size()) / 2);
    const auto global_start =
        static_cast<global_index_type>((rank * (rank + 1)) / 2);
    auto local_dim = gko::dim<2>{static_cast<gko::size_type>(local_size),
                                 static_cast<gko::size_type>(local_size)};
    auto diag = mtx_type::create(this->ref);
    gko::matrix_data<value_type, local_index_type> diag_data{
        local_dim, {{0, 0, value_type{1.0f}}}};
    gko::matrix_data<value_type, global_index_type> expected{
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)},
        {{global_start, global_start, value_type{1.0f}}}};
    if (local_size > 1) {
        diag_data.nonzeros.emplace_back(
            static_cast<local_index_type>(local_size - 1), 0, value_type{2.0f});
        expected.nonzeros.emplace_back(
            static_cast<global_index_type>(global_start + local_size - 1),
            global_start, value_type{2.0f});
    }
    diag->read(diag_data);
    auto matrix = dist_mtx_type::create(
        this->ref, this->comm,
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)},
        std::move(diag));
    gko::matrix_data<value_type, global_index_type> written;

    gko::as<writable_type>(matrix.get())->write(written);

    ASSERT_EQ(written.size, expected.size);
    ASSERT_EQ(written.nonzeros.size(), expected.nonzeros.size());
    for (gko::size_type i = 0; i < expected.nonzeros.size(); ++i) {
        EXPECT_EQ(written.nonzeros[i], expected.nonzeros[i]);
    }
}


TYPED_TEST(MatrixBuilder, ThrowsWhenWritingRectangularMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_mtx_type = gko::matrix::Csr<value_type, local_index_type>;
    using writable_type =
        gko::WritableToMatrixData<value_type, global_index_type>;
    const auto local_size =
        static_cast<local_index_type>(this->comm.rank() + 1);
    const auto global_size = static_cast<global_index_type>(
        ((1 + this->comm.size()) * this->comm.size()) / 2);
    auto diag = local_mtx_type::create(this->ref);
    diag->read(gko::matrix_data<value_type, local_index_type>{
        gko::dim<2>{static_cast<gko::size_type>(local_size),
                    static_cast<gko::size_type>(local_size)},
        {{0, 0, value_type{1.0f}}}});
    auto matrix = dist_mtx_type::create(
        this->ref, this->comm,
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size + 1)},
        std::move(diag));
    gko::matrix_data<value_type, global_index_type> written;

    ASSERT_THROW(gko::as<writable_type>(matrix.get())->write(written),
                 gko::DimensionMismatch);
}


}  // namespace

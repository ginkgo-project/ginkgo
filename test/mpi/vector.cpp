// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <random>


#include <mpi.h>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/log/logger.hpp>


#include "core/test/utils.hpp"
#include "test/utils/mpi/executor.hpp"


bool needs_transfers(std::shared_ptr<const gko::Executor> exec)
{
    return exec->get_master() != exec &&
           !gko::experimental::mpi::is_gpu_aware();
}


class HostToDeviceLogger : public gko::log::Logger {
public:
    void on_copy_started(const gko::Executor* exec_from,
                         const gko::Executor* exec_to,
                         const gko::uintptr& loc_from,
                         const gko::uintptr& loc_to,
                         const gko::size_type& num_bytes) const override
    {
        if (exec_from != exec_to) {
            transfer_count_++;
        }
    }

    int get_transfer_count() const { return transfer_count_; }

    static std::unique_ptr<HostToDeviceLogger> create()
    {
        return std::unique_ptr<HostToDeviceLogger>(new HostToDeviceLogger());
    }

protected:
    HostToDeviceLogger() : gko::log::Logger(gko::log::Logger::copy_started_mask)
    {}

private:
    mutable int transfer_count_ = 0;
};


template <typename ValueLocalGlobalIndexType>
class VectorCreation : public CommonMpiTestFixture {
public:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;
    using d_md_type = gko::device_matrix_data<value_type, global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;

    VectorCreation()
        : part(gko::share(part_type::build_from_contiguous(
              this->ref, {ref, {0, 2, 4, 6}}))),
          local_size{4, 11},
          size{local_size[1] * comm.size(), 11},
          md{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
          md_localized{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}}
    {}

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    std::shared_ptr<part_type> part;

    gko::dim<2> local_size;
    gko::dim<2> size;

    md_type md;
    md_type md_localized[3];

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(VectorCreation, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


#ifndef GKO_COMPILING_DPCPP


TYPED_TEST(VectorCreation, CanReadGlobalMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto vec = TestFixture::dist_vec_type::create(this->exec, this->comm);
    auto rank = this->comm.rank();
    I<I<value_type>> ref_data[3] = {
        {{0, 1}, {2, 3}},
        {{4, 5}, {6, 7}},
        {{8, 9}, {10, 11}},
    };

    vec->read_distributed(this->md, this->part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local_vector()->get_size(),
                                gko::dim<2>(2, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), ref_data[rank], 0.0);
}


TYPED_TEST(VectorCreation, CanReadGlobalMatrixDataSomeEmpty)
{
    using part_type = typename TestFixture::part_type;
    auto part = gko::share(part_type::build_from_contiguous(
        this->exec, {this->exec, {0, 0, 6, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->exec, this->comm);
    auto rank = this->comm.rank();

    vec->read_distributed(this->md, part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    if (rank == 1) {
        GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local_vector()->get_size(),
                                    gko::dim<2>(6, 2));
        GKO_ASSERT_MTX_NEAR(
            vec->get_local_vector(),
            l({{0., 1.}, {2., 3.}, {4., 5.}, {6., 7.}, {8., 9.}, {10., 11.}}),
            0.0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local_vector()->get_size(),
                                    gko::dim<2>(0, 2));
    }
}


TYPED_TEST(VectorCreation, CanReadGlobalDeviceMatrixData)
{
    using index_type = typename TestFixture::global_index_type;
    using d_md_type = typename TestFixture::d_md_type;
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    d_md_type md{
        this->exec, gko::dim<2>{6, 2},
        gko::array<index_type>{
            this->exec, I<index_type>{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}},
        gko::array<index_type>{
            this->exec, I<index_type>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
        gko::array<value_type>{
            this->exec, I<value_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}};
    auto part = gko::share(part_type::build_from_contiguous(
        this->exec, {this->exec, {0, 2, 4, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->exec, this->comm);
    auto rank = this->comm.rank();
    I<I<value_type>> ref_data[3] = {
        {{0, 1}, {2, 3}},
        {{4, 5}, {6, 7}},
        {{8, 9}, {10, 11}},
    };

    vec->read_distributed(md, part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local_vector()->get_size(),
                                gko::dim<2>(2, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), ref_data[rank], 0.0);
}


TYPED_TEST(VectorCreation, CanReadGlobalMatrixDataScattered)
{
    using md_type = typename TestFixture::md_type;
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    md_type md{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}};
    auto part = gko::share(part_type::build_from_mapping(
        this->exec, {this->exec, {0, 1, 2, 0, 2, 0}}, 3));
    auto vec = TestFixture::dist_vec_type::create(this->exec, this->comm);
    auto rank = this->comm.rank();
    gko::dim<2> ref_size[3] = {{3, 2}, {1, 2}, {2, 2}};
    I<I<value_type>> ref_data[3] = {
        {{0, 1}, {6, 7}, {10, 11}},
        {{2, 3}},
        {{4, 5}, {8, 9}},
    };

    vec->read_distributed(md, part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local_vector()->get_size(),
                                ref_size[rank]);
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), ref_data[rank], 0.0);
}


TYPED_TEST(VectorCreation, CanReadLocalMatrixData)
{
    using md_type = typename TestFixture::md_type;
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    md_type md[3] = {
        {gko::dim<2>{6, 2}, {{0, 0, 0}, {0, 1, 1}, {1, 0, 2}, {1, 1, 3}}},
        {gko::dim<2>{6, 2}, {{2, 0, 4}, {2, 1, 5}, {3, 0, 6}, {3, 1, 7}}},
        {gko::dim<2>{6, 2}, {{4, 0, 8}, {4, 1, 9}, {5, 0, 10}, {5, 1, 11}}}};
    auto part = gko::share(part_type::build_from_contiguous(
        this->exec, {this->exec, {0, 2, 4, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->exec, this->comm);
    auto rank = this->comm.rank();
    I<I<value_type>> ref_data[3] = {
        {{0, 1}, {2, 3}},
        {{4, 5}, {6, 7}},
        {{8, 9}, {10, 11}},
    };

    vec->read_distributed(md[rank], part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local_vector()->get_size(),
                                gko::dim<2>(2, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), ref_data[rank], 0.0);
}


TYPED_TEST(VectorCreation, CanReadLocalMatrixDataSomeEmpty)
{
    using md_type = typename TestFixture::md_type;
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    md_type md[3] = {{gko::dim<2>{6, 2}, {}},
                     {gko::dim<2>{6, 2},
                      // clang-format off
                      {{0, 0, 0}, {0, 1, 1},
                       {1, 0, 2}, {1, 1, 3},
                       {2, 0, 4}, {2, 1, 5},
                       {3, 0, 6}, {3, 1, 7},
                       {4, 0, 8}, {4, 1, 9},
                       {5, 0, 10}, {5, 1, 11}}},
                     // clang-format on
                     {gko::dim<2>{6, 2}, {}}};
    auto part = gko::share(part_type::build_from_contiguous(
        this->exec, {this->exec, {0, 0, 6, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->exec, this->comm);
    auto rank = this->comm.rank();

    vec->read_distributed(md[rank], part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    if (rank == 1) {
        GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local_vector()->get_size(),
                                    gko::dim<2>(6, 2));
        GKO_ASSERT_MTX_NEAR(
            vec->get_local_vector(),
            I<I<value_type>>(
                {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}}),
            0.0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local_vector()->get_size(),
                                    gko::dim<2>(0, 2));
    }
}


#endif


TYPED_TEST(VectorCreation, CanCreateFromLocalVectorAndSize)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using dense_type = typename TestFixture::dense_type;
    auto local_vec = dense_type::create(this->exec);
    local_vec->read(this->md_localized[this->comm.rank()]);

    auto vec = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{6, 2},
                                     gko::clone(local_vec));

    GKO_ASSERT_EQUAL_DIMENSIONS(vec, gko::dim<2>(6, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), local_vec, 0);
}


TYPED_TEST(VectorCreation, CanCreateFromLocalVectorWithoutSize)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using dense_type = typename TestFixture::dense_type;
    auto local_vec = dense_type::create(this->exec);
    local_vec->read(this->md_localized[this->comm.rank()]);

    auto vec =
        dist_vec_type::create(this->exec, this->comm, gko::clone(local_vec));

    GKO_ASSERT_EQUAL_DIMENSIONS(vec, gko::dim<2>(6, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), local_vec, 0);
}


TYPED_TEST(VectorCreation, CanConstCreateFromLocalVectorAndSize)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using dense_type = typename TestFixture::dense_type;
    auto local_vec = dense_type::create(this->exec);
    local_vec->read(this->md_localized[this->comm.rank()]);

    auto vec = dist_vec_type::create_const(
        this->exec, this->comm, gko::dim<2>{6, 2}, gko::clone(local_vec));

    ASSERT_TRUE(std::is_const<
                typename std::remove_reference<decltype(*vec)>::type>::value);
    GKO_ASSERT_EQUAL_DIMENSIONS(vec, gko::dim<2>(6, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), local_vec, 0);
}


TYPED_TEST(VectorCreation, CanConstCreateFromLocalVectorWithoutSize)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using dense_type = typename TestFixture::dense_type;
    auto local_vec = dense_type::create(this->exec);
    local_vec->read(this->md_localized[this->comm.rank()]);

    auto vec = dist_vec_type::create_const(this->exec, this->comm,
                                           gko::clone(local_vec));

    ASSERT_TRUE(std::is_const<
                typename std::remove_reference<decltype(*vec)>::type>::value);
    GKO_ASSERT_EQUAL_DIMENSIONS(vec, gko::dim<2>(6, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), local_vec, 0);
}

template <typename ValueType>
class VectorCreationHelpers : public CommonMpiTestFixture {
public:
    using value_type = ValueType;
    using vec_type = gko::experimental::distributed::Vector<value_type>;

    VectorCreationHelpers()
        : local_size{4, 11},
          size{local_size[1] * comm.size(), 11},
          src(vec_type::create(this->exec, this->comm, size, local_size,
                               local_size[1] + 3))
    {}

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    gko::dim<2> local_size;
    gko::dim<2> size;

    std::unique_ptr<vec_type> src;
    std::unique_ptr<vec_type> dst;
};

TYPED_TEST_SUITE(VectorCreationHelpers, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(VectorCreationHelpers, CanCreateWithConfigOf)
{
    using vec_type = typename TestFixture::vec_type;

    auto new_vector = vec_type::create_with_config_of(this->src);

    GKO_ASSERT_EQUAL_DIMENSIONS(new_vector->get_size(), this->src->get_size());
    GKO_ASSERT_EQUAL_DIMENSIONS(new_vector->get_local_vector()->get_size(),
                                this->src->get_local_vector()->get_size());
    ASSERT_EQ(new_vector->get_local_vector()->get_stride(),
              this->src->get_local_vector()->get_stride());
    ASSERT_EQ(new_vector->get_executor(), this->src->get_executor());
    ASSERT_EQ(new_vector->get_communicator(), this->src->get_communicator());
}


TYPED_TEST(VectorCreationHelpers, CanCreateWithTypeOfDefaultParameter)
{
    using vec_type = typename TestFixture::vec_type;

    auto new_vector = vec_type::create_with_type_of(this->src, this->ref);

    GKO_ASSERT_EQUAL_DIMENSIONS(new_vector->get_size(), gko::dim<2>{});
    GKO_ASSERT_EQUAL_DIMENSIONS(new_vector->get_local_vector()->get_size(),
                                gko::dim<2>{});
    ASSERT_EQ(new_vector->get_local_vector()->get_stride(), 0);
    ASSERT_EQ(new_vector->get_executor(), this->ref);
    ASSERT_EQ(new_vector->get_communicator(), this->src->get_communicator());
}


TYPED_TEST(VectorCreationHelpers, CanCreateWithTypeOf)
{
    using vec_type = typename TestFixture::vec_type;
    gko::dim<2> new_local_size{3, 7};
    gko::dim<2> new_global_size{new_local_size[0] * this->comm.size(),
                                new_local_size[1]};
    gko::size_type new_stride{14};

    auto new_vector = vec_type::create_with_type_of(
        this->src, this->ref, new_global_size, new_local_size, new_stride);

    GKO_ASSERT_EQUAL_DIMENSIONS(new_vector->get_size(), new_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(new_vector->get_local_vector()->get_size(),
                                new_local_size);
    ASSERT_EQ(new_vector->get_local_vector()->get_stride(), new_stride);
    ASSERT_EQ(new_vector->get_executor(), this->ref);
    ASSERT_EQ(new_vector->get_communicator(), this->src->get_communicator());
}


template <typename ValueType>
class VectorReductions : public CommonMpiTestFixture {
public:
    using value_type = ValueType;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;
    using real_dense_type = typename dense_type::real_type;

    VectorReductions() : size{53, 11}, engine(42)
    {
        logger = gko::share(HostToDeviceLogger::create());
        exec->add_logger(logger);

        dense_x = dense_type::create(exec);
        dense_y = dense_type::create(exec);
        x = dist_vec_type::create(exec, comm);
        y = dist_vec_type::create(exec, comm);
        dense_res = dense_type ::create(exec);
        res = dense_type ::create(exec);
        dense_real_res = real_dense_type ::create(exec);
        real_res = real_dense_type ::create(exec);

        dense_tmp = gko::Array<char>(exec);
        tmp = gko::Array<char>(exec);

        auto num_parts =
            static_cast<gko::experimental::distributed::comm_index_type>(
                comm.size());
        auto mapping = gko::test::generate_random_array<
            gko::experimental::distributed::comm_index_type>(
            size[0],
            std::uniform_int_distribution<
                gko::experimental::distributed::comm_index_type>(0,
                                                                 num_parts - 1),
            engine, ref);
        auto part = part_type::build_from_mapping(ref, mapping, num_parts);

        auto md_x = gko::test::generate_random_matrix_data<value_type,
                                                           global_index_type>(
            size[0], size[1],
            std::uniform_int_distribution<gko::size_type>(size[1], size[1]),
            std::normal_distribution<gko::remove_complex<value_type>>(),
            engine);
        dense_x->read(md_x);
        auto tmp_x = dist_vec_type::create(ref, comm);
        tmp_x->read_distributed(md_x, part);
        x = gko::clone(exec, tmp_x);

        auto md_y = gko::test::generate_random_matrix_data<value_type,
                                                           global_index_type>(
            size[0], size[1],
            std::uniform_int_distribution<gko::size_type>(size[1], size[1]),
            std::normal_distribution<gko::remove_complex<value_type>>(),
            engine);
        dense_y->read(md_y);
        auto tmp_y = dist_vec_type::create(ref, comm);
        tmp_y->read_distributed(md_y, part);
        y = gko::clone(exec, tmp_y);
    }

    void SetUp() override { ASSERT_GT(comm.size(), 0); }

    void init_result()
    {
        res = dense_type::create(exec, gko::dim<2>{1, size[1]});
        dense_res = dense_type::create(exec, gko::dim<2>{1, size[1]});
        real_res = real_dense_type::create(exec, gko::dim<2>{1, size[1]});
        dense_real_res = real_dense_type::create(exec, gko::dim<2>{1, size[1]});
        res->fill(0.0);
        dense_res->fill(0.0);
        real_res->fill(0.0);
        dense_real_res->fill(0.0);
    }

    gko::dim<2> size;

    std::unique_ptr<dense_type> dense_x;
    std::unique_ptr<dense_type> dense_y;
    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> y;
    std::unique_ptr<dense_type> dense_res;
    std::unique_ptr<dense_type> res;
    std::unique_ptr<real_dense_type> dense_real_res;
    std::unique_ptr<real_dense_type> real_res;
    gko::array<char> dense_tmp;
    gko::array<char> tmp;

    std::shared_ptr<HostToDeviceLogger> logger;

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(VectorReductions, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(VectorReductions, ComputeDotProductIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_dot(this->y, this->res);
    this->dense_x->compute_dot(this->dense_y, this->dense_res);

    GKO_ASSERT_MTX_NEAR(this->res, this->dense_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeDotProductWithTmpIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_dot(this->y, this->res, this->tmp);
    this->dense_x->compute_dot(this->dense_y, this->dense_res, this->dense_tmp);

    GKO_ASSERT_MTX_NEAR(this->res, this->dense_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeConjDotProductIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_conj_dot(this->y, this->res);
    this->dense_x->compute_conj_dot(this->dense_y, this->dense_res);

    GKO_ASSERT_MTX_NEAR(this->res, this->dense_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeConjDotProductWithTmpIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_conj_dot(this->y, this->res, this->tmp);
    this->dense_x->compute_conj_dot(this->dense_y, this->dense_res,
                                    this->dense_tmp);

    GKO_ASSERT_MTX_NEAR(this->res, this->dense_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeNorm2IsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_norm2(this->real_res);
    this->dense_x->compute_norm2(this->dense_real_res);

    GKO_ASSERT_MTX_NEAR(this->real_res, this->dense_real_res,
                        r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeNorm2WithTmpIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_norm2(this->real_res, this->tmp);
    this->dense_x->compute_norm2(this->dense_real_res, this->dense_tmp);

    GKO_ASSERT_MTX_NEAR(this->real_res, this->dense_real_res,
                        r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeNorm1IsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_norm1(this->real_res);
    this->dense_x->compute_norm1(this->dense_real_res);

    GKO_ASSERT_MTX_NEAR(this->real_res, this->dense_real_res,
                        r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeNorm1WithTmpIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_norm1(this->real_res, this->tmp);
    this->dense_x->compute_norm1(this->dense_real_res, this->dense_tmp);

    GKO_ASSERT_MTX_NEAR(this->real_res, this->dense_real_res,
                        r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeSquaredNorm2IsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_squared_norm2(this->real_res);
    this->dense_x->compute_squared_norm2(this->dense_real_res);

    GKO_ASSERT_MTX_NEAR(this->real_res, this->dense_real_res,
                        r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeSquaredNorm2WithTmpIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_squared_norm2(this->real_res, this->tmp);
    this->dense_x->compute_squared_norm2(this->dense_real_res, this->dense_tmp);

    GKO_ASSERT_MTX_NEAR(this->real_res, this->dense_real_res,
                        r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputesMeanIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_mean(this->res);
    this->dense_x->compute_mean(this->dense_res);

    GKO_ASSERT_MTX_NEAR(this->res, this->dense_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputesMeanWithTmpIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_mean(this->res, this->tmp);
    this->dense_x->compute_mean(this->dense_res, this->dense_tmp);

    GKO_ASSERT_MTX_NEAR(this->res, this->dense_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeDotCopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_dot(this->y, this->res);

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


TYPED_TEST(VectorReductions, ComputeConjDotCopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_conj_dot(this->y, this->res);

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


TYPED_TEST(VectorReductions, ComputeNorm2CopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_norm2(this->real_res);

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


TYPED_TEST(VectorReductions, ComputeNorm1CopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_norm1(this->real_res);

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


TYPED_TEST(VectorReductions, ComputeSquaredNorm2CopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_squared_norm2(this->real_res);

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


template <typename ValueType>
class VectorLocalOps : public CommonMpiTestFixture {
public:
    using value_type = ValueType;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using complex_dist_vec_type = typename dist_vec_type::complex_type;
    using real_dist_vec_type = typename dist_vec_type ::real_type;
    using dense_type = gko::matrix::Dense<value_type>;
    using complex_dense_type = typename dense_type::complex_type;
    using real_dense_type = typename dense_type ::real_type;

    VectorLocalOps()
        : local_size{4, 11}, size{local_size[0] * comm.size(), 11}, engine(42)
    {
        x = dist_vec_type::create(exec, comm);
        y = dist_vec_type::create(exec, comm);
        alpha = dense_type ::create(exec);
        local_complex = complex_dense_type ::create(exec);
        complex = complex_dist_vec_type::create(exec, comm);
    }

    void SetUp() override { ASSERT_GT(comm.size(), 0); }

    template <typename LocalVectorType, typename DistVectorType>
    void generate_vector_pair(std::unique_ptr<LocalVectorType>& local,
                              std::unique_ptr<DistVectorType>& dist)
    {
        using vtype = typename LocalVectorType::value_type;
        local = gko::test::generate_random_matrix<LocalVectorType>(
            local_size[0], local_size[1],
            std::uniform_int_distribution<gko::size_type>(local_size[1],
                                                          local_size[1]),
            std::normal_distribution<gko::remove_complex<vtype>>(), engine,
            exec);
        dist = DistVectorType::create(exec, comm, size, gko::clone(local));
    }

    void init_vectors()
    {
        generate_vector_pair(local_x, x);
        generate_vector_pair(local_y, y);

        alpha = gko::test::generate_random_matrix<dense_type>(
            1, size[1],
            std::uniform_int_distribution<gko::size_type>(size[1], size[1]),
            std::normal_distribution<gko::remove_complex<value_type>>(), engine,
            exec);
    }

    void init_complex_vectors()
    {
        generate_vector_pair(local_real, real);
        generate_vector_pair(local_complex, complex);
    }

    gko::dim<2> local_size;
    gko::dim<2> size;

    std::unique_ptr<dense_type> local_x;
    std::unique_ptr<dense_type> local_y;
    std::unique_ptr<complex_dense_type> local_complex;
    std::unique_ptr<real_dense_type> local_real;
    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> y;
    std::unique_ptr<dense_type> alpha;
    std::unique_ptr<complex_dist_vec_type> complex;
    std::unique_ptr<real_dist_vec_type> real;

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(VectorLocalOps, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(VectorLocalOps, ApplyNotSupported)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto a = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto b = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto c = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});

    ASSERT_THROW(a->apply(b, c), gko::NotSupported);
}


TYPED_TEST(VectorLocalOps, AdvancedApplyNotSupported)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto a = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto b = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{1, 1},
                                   gko::dim<2>{1, 1});
    auto c = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto d = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{1, 1},
                                   gko::dim<2>{1, 1});
    auto e = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});

    ASSERT_THROW(a->apply(b, c, d, e), gko::NotSupported);
}


TYPED_TEST(VectorLocalOps, ConvertsToPrecision)
{
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherVector = typename gko::experimental::distributed::Vector<OtherT>;
    auto local_tmp = OtherVector::local_vector_type::create(this->exec);
    auto tmp = OtherVector::create(this->exec, this->comm);
    this->init_vectors();

    this->local_x->convert_to(local_tmp);
    this->x->convert_to(tmp);

    GKO_ASSERT_MTX_NEAR(tmp->get_local_vector(), local_tmp, 0.0);
}


TYPED_TEST(VectorLocalOps, MovesToPrecision)
{
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherVector = typename gko::experimental::distributed::Vector<OtherT>;
    auto local_tmp = OtherVector::local_vector_type::create(this->exec);
    auto tmp = OtherVector::create(this->exec, this->comm);
    this->init_vectors();

    this->local_x->move_to(local_tmp);
    this->x->move_to(tmp);

    GKO_ASSERT_MTX_NEAR(tmp->get_local_vector(), local_tmp, 0.0);
}


TYPED_TEST(VectorLocalOps, ComputeAbsoluteSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    auto local_abs = this->local_x->compute_absolute();
    auto abs = this->x->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs->get_local_vector(), local_abs,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, ComputeAbsoluteInplaceSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    this->local_x->compute_absolute_inplace();
    this->x->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, MakeComplexSameAsLocal)
{
    this->init_vectors();
    this->init_complex_vectors();

    this->complex = this->x->make_complex();
    this->local_complex = this->local_x->make_complex();

    GKO_ASSERT_MTX_NEAR(this->complex->get_local_vector(), this->local_complex,
                        0.0);
}


TYPED_TEST(VectorLocalOps, MakeComplexInplaceSameAsLocal)
{
    this->init_vectors();
    this->init_complex_vectors();

    this->x->make_complex(this->complex);
    this->local_x->make_complex(this->local_complex);

    GKO_ASSERT_MTX_NEAR(this->complex->get_local_vector(), this->local_complex,
                        0.0);
}


TYPED_TEST(VectorLocalOps, GetRealSameAsLocal)
{
    this->init_vectors();
    this->init_complex_vectors();

    this->real = this->complex->get_real();
    this->local_real = this->local_complex->get_real();

    GKO_ASSERT_MTX_NEAR(this->real->get_local_vector(), this->local_real, 0.0);
}


TYPED_TEST(VectorLocalOps, GetRealInplaceSameAsLocal)
{
    this->init_vectors();
    this->init_complex_vectors();

    this->complex->get_real(this->real);
    this->local_complex->get_real(this->local_real);

    GKO_ASSERT_MTX_NEAR(this->real->get_local_vector(), this->local_real, 0.0);
}


TYPED_TEST(VectorLocalOps, GetImagSameAsLocal)
{
    this->init_complex_vectors();

    this->real = this->complex->get_imag();
    this->local_real = this->local_complex->get_imag();

    GKO_ASSERT_MTX_NEAR(this->real->get_local_vector(), this->local_real, 0.0);
}


TYPED_TEST(VectorLocalOps, GetImagInplaceSameAsLocal)
{
    this->init_complex_vectors();

    this->complex->get_imag(this->real);
    this->local_complex->get_imag(this->local_real);

    GKO_ASSERT_MTX_NEAR(this->real->get_local_vector(), this->local_real, 0.0);
}


TYPED_TEST(VectorLocalOps, FillSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    auto value = gko::test::detail::get_rand_value<value_type>(
        std::normal_distribution<gko::remove_complex<value_type>>(),
        this->engine);
    this->init_vectors();

    this->x->fill(value);
    this->local_x->fill(value);

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x, 0.0);
}


TYPED_TEST(VectorLocalOps, ScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    this->x->scale(this->alpha);
    this->local_x->scale(this->alpha);

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, InvScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    this->x->inv_scale(this->alpha);
    this->local_x->inv_scale(this->alpha);

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, AddScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    this->x->add_scaled(this->alpha, this->y);
    this->local_x->add_scaled(this->alpha, this->local_y);

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, SubScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    this->x->sub_scaled(this->alpha, this->y);
    this->local_x->sub_scaled(this->alpha, this->local_y);

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, CreateRealViewSameAsLocal)
{
    this->init_vectors();

    auto rv = this->x->create_real_view();
    auto local_rv = this->local_x->create_real_view();

    GKO_ASSERT_EQUAL_ROWS(rv, this->x);
    GKO_ASSERT_EQUAL_ROWS(rv->get_local_vector(), local_rv);
    GKO_ASSERT_EQUAL_COLS(rv->get_local_vector(), local_rv);
    EXPECT_EQ(rv->get_local_vector()->get_stride(), local_rv->get_stride());
    GKO_ASSERT_MTX_NEAR(rv->get_local_vector(), local_rv, 0.0);
}

/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <mpi.h>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/log/logger.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


bool needs_transfers(std::shared_ptr<const gko::Executor> exec)
{
    return exec->get_master() != exec && !gko::mpi::is_gpu_aware();
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

    static std::unique_ptr<HostToDeviceLogger> create(
        std::shared_ptr<const gko::Executor> exec)
    {
        return std::unique_ptr<HostToDeviceLogger>(
            new HostToDeviceLogger(std::move(exec)));
    }

protected:
    explicit HostToDeviceLogger(std::shared_ptr<const gko::Executor> exec)
        : gko::log::Logger(exec, gko::log::Logger::copy_started_mask)
    {}

private:
    mutable int transfer_count_ = 0;
};


template <typename ValueLocalGlobalIndexType>
class VectorCreation : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using local_index_type =
        typename std::tuple_element<1, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using global_index_type =
        typename std::tuple_element<2, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;
    using d_md_type = gko::device_matrix_data<value_type, global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;

    VectorCreation()
        : ref(gko::ReferenceExecutor::create()),
          comm(MPI_COMM_WORLD),
          part(gko::share(part_type::build_from_contiguous(
              this->ref, {ref, {0, 2, 4, 6}}))),
          local_size{4, 11},
          size{local_size[1] * comm.size(), 11},
          md{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
          md_localized{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}}
    {}

    void SetUp() override
    {
        ASSERT_EQ(this->comm.size(), 3);
        init_executor(gko::ReferenceExecutor::create(), exec);
    }

    void TearDown() override
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    gko::mpi::communicator comm;
    std::shared_ptr<part_type> part;

    gko::dim<2> local_size;
    gko::dim<2> size;

    md_type md;
    md_type md_localized[3];

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(VectorCreation, gko::test::ValueLocalGlobalIndexTypes);


#ifdef GKO_COMPILING_REFERENCE


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

    vec->read_distributed(this->md, this->part.get());

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

    vec->read_distributed(this->md, part.get());

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
        gko::Array<index_type>{
            this->exec, I<index_type>{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}},
        gko::Array<index_type>{
            this->exec, I<index_type>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
        gko::Array<value_type>{
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

    vec->read_distributed(md, part.get());

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

    vec->read_distributed(md, part.get());

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

    vec->read_distributed(md[rank], part.get());

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

    vec->read_distributed(md[rank], part.get());

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
    auto clone_local_vec = gko::clone(local_vec);

    auto vec = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{6, 2},
                                     local_vec.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec, gko::dim<2>(6, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), clone_local_vec, 0);
}


TYPED_TEST(VectorCreation, CanCreateFromLocalVectorWithoutSize)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using dense_type = typename TestFixture::dense_type;
    auto local_vec = dense_type::create(this->exec);
    local_vec->read(this->md_localized[this->comm.rank()]);
    auto clone_local_vec = gko::clone(local_vec);

    auto vec = dist_vec_type::create(this->exec, this->comm, local_vec.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec, gko::dim<2>(6, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local_vector(), clone_local_vec, 0);
}


template <typename ValueType>
class VectorReductions : public ::testing::Test {
public:
    using value_type = ValueType;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;
    using real_dense_type = typename dense_type::real_type;

    VectorReductions()
        : ref(gko::ReferenceExecutor::create()),
          exec(),
          comm(MPI_COMM_WORLD),
          size{53, 11},
          engine(42)
    {
        init_executor(ref, exec, comm);

        logger = gko::share(HostToDeviceLogger::create(exec));
        exec->add_logger(logger);

        dense_x = dense_type::create(exec);
        dense_y = dense_type::create(exec);
        x = dist_vec_type::create(exec, comm);
        y = dist_vec_type::create(exec, comm);
        dense_res = dense_type ::create(exec);
        res = dense_type ::create(exec);
        dense_real_res = real_dense_type ::create(exec);
        real_res = real_dense_type ::create(exec);

        auto num_parts =
            static_cast<gko::distributed::comm_index_type>(comm.size());
        auto mapping =
            gko::test::generate_random_array<gko::distributed::comm_index_type>(
                size[0],
                std::uniform_int_distribution<
                    gko::distributed::comm_index_type>(0, num_parts - 1),
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
        tmp_x->read_distributed(md_x, part.get());
        x = gko::clone(exec, tmp_x);

        auto md_y = gko::test::generate_random_matrix_data<value_type,
                                                           global_index_type>(
            size[0], size[1],
            std::uniform_int_distribution<gko::size_type>(size[1], size[1]),
            std::normal_distribution<gko::remove_complex<value_type>>(),
            engine);
        dense_y->read(md_y);
        auto tmp_y = dist_vec_type::create(ref, comm);
        tmp_y->read_distributed(md_y, part.get());
        y = gko::clone(exec, tmp_y);
    }

    void SetUp() override
    {
        ASSERT_GT(comm.size(), 0);
        init_executor(gko::ReferenceExecutor::create(), exec);
    }

    void TearDown() override
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

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

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    gko::mpi::communicator comm;

    gko::dim<2> size;

    std::unique_ptr<dense_type> dense_x;
    std::unique_ptr<dense_type> dense_y;
    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> y;
    std::unique_ptr<dense_type> dense_res;
    std::unique_ptr<dense_type> res;
    std::unique_ptr<real_dense_type> dense_real_res;
    std::unique_ptr<real_dense_type> real_res;

    std::shared_ptr<HostToDeviceLogger> logger;

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(VectorReductions, gko::test::ValueTypes);


TYPED_TEST(VectorReductions, ComputesDotProductIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_dot(this->y.get(), this->res.get());
    this->dense_x->compute_dot(this->dense_y.get(), this->dense_res.get());

    GKO_ASSERT_MTX_NEAR(this->res, this->dense_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputesConjDotProductIsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_conj_dot(this->y.get(), this->res.get());
    this->dense_x->compute_conj_dot(this->dense_y.get(), this->dense_res.get());

    GKO_ASSERT_MTX_NEAR(this->res, this->dense_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputesNorm2IsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_norm2(this->real_res.get());
    this->dense_x->compute_norm2(this->dense_real_res.get());

    GKO_ASSERT_MTX_NEAR(this->real_res, this->dense_real_res,
                        r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputesNorm1IsSameAsDense)
{
    using value_type = typename TestFixture::value_type;
    this->init_result();

    this->x->compute_norm1(this->real_res.get());
    this->dense_x->compute_norm1(this->dense_real_res.get());

    GKO_ASSERT_MTX_NEAR(this->real_res, this->dense_real_res,
                        r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputeDotCopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_dot(this->y.get(), this->res.get());

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


TYPED_TEST(VectorReductions, ComputeConjDotCopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_conj_dot(this->y.get(), this->res.get());

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


TYPED_TEST(VectorReductions, ComputeNorm2CopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_norm2(this->real_res.get());

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


TYPED_TEST(VectorReductions, ComputeNorm1CopiesToHostOnlyIfNecessary)
{
    this->init_result();
    auto transfer_count_before = this->logger->get_transfer_count();

    this->x->compute_norm1(this->real_res.get());

    ASSERT_EQ(this->logger->get_transfer_count() > transfer_count_before,
              needs_transfers(this->exec));
}


template <typename ValueType>
class VectorLocalOps : public ::testing::Test {
public:
    using value_type = ValueType;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using complex_dist_vec_type = typename dist_vec_type::complex_type;
    using real_dist_vec_type = typename dist_vec_type ::real_type;
    using dense_type = gko::matrix::Dense<value_type>;
    using complex_dense_type = typename dense_type::complex_type;
    using real_dense_type = typename dense_type ::real_type;

    VectorLocalOps()
        : ref(gko::ReferenceExecutor::create()),
          exec(),
          comm(MPI_COMM_WORLD),
          local_size{4, 11},
          size{local_size[0] * comm.size(), 11},
          engine(42)
    {
        init_executor(ref, exec, comm);

        x = dist_vec_type::create(exec, comm);
        y = dist_vec_type::create(exec, comm);
        alpha = dense_type ::create(exec);
        local_complex = complex_dense_type ::create(exec);
        complex = complex_dist_vec_type::create(exec, comm);
    }

    void SetUp() override
    {
        ASSERT_GT(comm.size(), 0);
        init_executor(gko::ReferenceExecutor::create(), exec);
    }

    void TearDown() override
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

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
        dist =
            DistVectorType::create(exec, comm, size, gko::clone(local).get());
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

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    gko::mpi::communicator comm;

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

TYPED_TEST_SUITE(VectorLocalOps, gko::test::ValueTypes);


TYPED_TEST(VectorLocalOps, ApplyNotSupported)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto a = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto b = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto c = dist_vec_type::create(this->exec, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});

    ASSERT_THROW(a->apply(b.get(), c.get()), gko::NotSupported);
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

    ASSERT_THROW(a->apply(b.get(), c.get(), d.get(), e.get()),
                 gko::NotSupported);
}


TYPED_TEST(VectorLocalOps, ConvertsToPrecision)
{
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherVector = typename gko::distributed::Vector<OtherT>;
    auto local_tmp = OtherVector::local_vector_type::create(this->exec);
    auto tmp = OtherVector::create(this->exec, this->comm);
    this->init_vectors();

    this->local_x->convert_to(local_tmp.get());
    this->x->convert_to(tmp.get());

    GKO_ASSERT_MTX_NEAR(tmp->get_local_vector(), local_tmp, 0.0);
}


TYPED_TEST(VectorLocalOps, MovesToPrecision)
{
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherVector = typename gko::distributed::Vector<OtherT>;
    auto local_tmp = OtherVector::local_vector_type::create(this->exec);
    auto tmp = OtherVector::create(this->exec, this->comm);
    this->init_vectors();

    this->local_x->move_to(local_tmp.get());
    this->x->move_to(tmp.get());

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

    this->x->make_complex(this->complex.get());
    this->local_x->make_complex(this->local_complex.get());

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

    this->complex->get_real(this->real.get());
    this->local_complex->get_real(this->local_real.get());

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

    this->complex->get_imag(this->real.get());
    this->local_complex->get_imag(this->local_real.get());

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

    this->x->scale(this->alpha.get());
    this->local_x->scale(this->alpha.get());

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, InvScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    this->x->inv_scale(this->alpha.get());
    this->local_x->inv_scale(this->alpha.get());

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, AddScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    this->x->add_scaled(this->alpha.get(), this->y.get());
    this->local_x->add_scaled(this->alpha.get(), this->local_y.get());

    GKO_ASSERT_MTX_NEAR(this->x->get_local_vector(), this->local_x,
                        r<value_type>::value);
}


TYPED_TEST(VectorLocalOps, SubScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    this->init_vectors();

    this->x->sub_scaled(this->alpha.get(), this->y.get());
    this->local_x->sub_scaled(this->alpha.get(), this->local_y.get());

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


}  // namespace

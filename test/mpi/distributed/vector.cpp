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


class VectorCreation : public ::testing::Test {
public:
    using value_type = float;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using dense_type = dist_vec_type::local_vector_type;

    VectorCreation()
        : ref(gko::ReferenceExecutor::create()),
          exec(),
          comm(MPI_COMM_WORLD),
          local_size{4, 11},
          size{local_size[1] * comm.size(), 11},
          engine(42)
    {
        init_executor(ref, exec, comm);
    }

    void SetUp() override { ASSERT_GT(comm.size(), 0); }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    gko::mpi::communicator comm;

    gko::dim<2> local_size;
    gko::dim<2> size;

    std::default_random_engine engine;
};


TEST_F(VectorCreation, CanCreateFromLocalVectorAndSize)
{
    auto local_vec = gko::test::generate_random_matrix<dense_type>(
        local_size[0], local_size[1],
        std::uniform_int_distribution<gko::size_type>(0, local_size[1]),
        std::normal_distribution<value_type>(), engine, ref);
    auto dlocal_vec = gko::clone(exec, local_vec);

    auto vec = dist_vec_type::create(ref, comm, size, local_vec.get());
    auto dvec = dist_vec_type::create(exec, comm, size, local_vec.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec, dvec);
    GKO_ASSERT_MTX_NEAR(vec->get_local(), dvec->get_local(), 0);
}


TEST_F(VectorCreation, CanCreateFromLocalVectorWithoutSize)
{
    auto local_vec = gko::test::generate_random_matrix<dense_type>(
        local_size[0], local_size[1],
        std::uniform_int_distribution<gko::size_type>(0, local_size[1]),
        std::normal_distribution<value_type>(), engine, ref);
    auto dlocal_vec = gko::clone(exec, local_vec);

    auto vec = dist_vec_type::create(ref, comm, local_vec.get());
    auto dvec = dist_vec_type::create(exec, comm, local_vec.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec, dvec);
    GKO_ASSERT_MTX_NEAR(vec->get_local(), dvec->get_local(), 0);
}


class VectorReductions : public ::testing::Test {
public:
    using value_type = float;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;

    VectorReductions()
        : ref(gko::ReferenceExecutor::create()),
          exec(),
          comm(MPI_COMM_WORLD),
          size{53, 11},
          x(dist_vec_type::create(ref, comm)),
          dx(dist_vec_type::create(exec, comm)),
          y(dist_vec_type::create(ref, comm)),
          dy(dist_vec_type::create(exec, comm)),
          logger(gko::share(HostToDeviceLogger::create(exec))),
          engine(42)
    {
        init_executor(ref, exec, comm);
        exec->add_logger(logger);

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
        x->read_distributed(md_x, part.get());
        dx = gko::clone(exec, x);

        auto md_y = gko::test::generate_random_matrix_data<value_type,
                                                           global_index_type>(
            size[0], size[1],
            std::uniform_int_distribution<gko::size_type>(size[1], size[1]),
            std::normal_distribution<gko::remove_complex<value_type>>(),
            engine);
        y->read_distributed(md_y, part.get());
        dy = gko::clone(exec, y);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    void init_result()
    {
        res = dense_type::create(ref, gko::dim<2>{1, size[1]});
        dres = dense_type::create(exec, gko::dim<2>{1, size[1]});
        res->fill(0.);
        dres->fill(0.);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    gko::mpi::communicator comm;

    gko::dim<2> size;

    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> dx;
    std::unique_ptr<dist_vec_type> y;
    std::unique_ptr<dist_vec_type> dy;
    std::unique_ptr<dense_type> res;
    std::unique_ptr<dense_type> dres;

    std::shared_ptr<HostToDeviceLogger> logger;

    std::default_random_engine engine;
};


TEST_F(VectorReductions, ComputesDotProductIsSameAsRef)
{
    init_result();

    x->compute_dot(y.get(), res.get());
    dx->compute_dot(dy.get(), dres.get());

    GKO_ASSERT_MTX_NEAR(res, dres, r<value_type>::value);
}


TEST_F(VectorReductions, ComputesConjDotProductIsSameAsRef)
{
    init_result();

    x->compute_conj_dot(y.get(), res.get());
    dx->compute_conj_dot(dy.get(), dres.get());

    GKO_ASSERT_MTX_NEAR(res, dres, r<value_type>::value);
}


TEST_F(VectorReductions, ComputesNorm2IsSameAsRef)
{
    init_result();

    x->compute_norm2(res.get());
    dx->compute_norm2(dres.get());

    GKO_ASSERT_MTX_NEAR(res, dres, r<value_type>::value);
}


TEST_F(VectorReductions, ComputesNorm1IsSameAsRef)
{
    init_result();

    x->compute_norm1(res.get());
    dx->compute_norm1(dres.get());

    GKO_ASSERT_MTX_NEAR(res, dres, r<value_type>::value);
}


TEST_F(VectorReductions, ComputeDotCopiesToHostOnlyIfNecessary)
{
    init_result();
    auto transfer_count_before = logger->get_transfer_count();

    dx->compute_dot(dy.get(), dres.get());

    ASSERT_EQ(logger->get_transfer_count() > transfer_count_before,
              needs_transfers(exec));
}


TEST_F(VectorReductions, ComputeConjDotCopiesToHostOnlyIfNecessary)
{
    init_result();
    auto transfer_count_before = logger->get_transfer_count();

    dx->compute_conj_dot(dy.get(), dres.get());

    ASSERT_EQ(logger->get_transfer_count() > transfer_count_before,
              needs_transfers(exec));
}


TEST_F(VectorReductions, ComputeNorm2CopiesToHostOnlyIfNecessary)
{
    init_result();
    auto transfer_count_before = logger->get_transfer_count();

    dx->compute_norm2(dres.get());

    ASSERT_EQ(logger->get_transfer_count() > transfer_count_before,
              needs_transfers(exec));
}


TEST_F(VectorReductions, ComputeNorm1CopiesToHostOnlyIfNecessary)
{
    init_result();
    auto transfer_count_before = logger->get_transfer_count();

    dx->compute_norm1(dres.get());

    ASSERT_EQ(logger->get_transfer_count() > transfer_count_before,
              needs_transfers(exec));
}


class VectorLocalOps : public ::testing::Test {
public:
    using value_type = float;
    using mixed_type = double;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using complex_dist_vec_type =
        gko::distributed::Vector<value_type>::complex_type;
    using dense_type = gko::matrix::Dense<value_type>;

    VectorLocalOps()
        : ref(gko::ReferenceExecutor::create()),
          exec(),
          comm(MPI_COMM_WORLD),
          size{53, 11},
          engine(42)
    {
        init_executor(ref, exec, comm);

        x = dist_vec_type::create(ref, comm);
        dx = dist_vec_type::create(exec, comm);
        y = dist_vec_type::create(ref, comm);
        dy = dist_vec_type::create(exec, comm);
        alpha = dense_type ::create(ref);
        dalpha = dense_type ::create(exec);
        complex = complex_dist_vec_type::create(ref, comm);
        dcomplex = complex_dist_vec_type::create(exec, comm);

        auto num_parts =
            static_cast<gko::distributed::comm_index_type>(comm.size());
        auto mapping =
            gko::test::generate_random_array<gko::distributed::comm_index_type>(
                size[0],
                std::uniform_int_distribution<
                    gko::distributed::comm_index_type>(0, num_parts - 1),
                engine, ref);
        part = part_type::build_from_mapping(ref, mapping, num_parts);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    template <typename VectorType>
    void generate_vector_pair(std::unique_ptr<VectorType>& host,
                              std::unique_ptr<VectorType>& device)
    {
        using vtype = typename VectorType::value_type;
        auto md =
            gko::test::generate_random_matrix_data<vtype, global_index_type>(
                size[0], size[1],
                std::uniform_int_distribution<gko::size_type>(size[1], size[1]),
                std::normal_distribution<gko::remove_complex<vtype>>(), engine);
        host->read_distributed(md, part.get());
        device = gko::clone(exec, host);
    }

    void init_vectors()
    {
        generate_vector_pair(x, dx);
        generate_vector_pair(y, dy);

        alpha = gko::test::generate_random_matrix<dense_type>(
            1, size[1],
            std::uniform_int_distribution<gko::size_type>(size[1], size[1]),
            std::normal_distribution<value_type>(), engine, ref);
        dalpha = gko::clone(exec, alpha);
    }

    void init_complex_vectors() { generate_vector_pair(complex, dcomplex); }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    gko::mpi::communicator comm;

    gko::dim<2> size;

    std::unique_ptr<part_type> part;

    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> dx;
    std::unique_ptr<dist_vec_type> y;
    std::unique_ptr<dist_vec_type> dy;
    std::unique_ptr<dense_type> alpha;
    std::unique_ptr<dense_type> dalpha;
    std::unique_ptr<complex_dist_vec_type> complex;
    std::unique_ptr<complex_dist_vec_type> dcomplex;

    std::default_random_engine engine;
};


TEST_F(VectorLocalOps, ConvertsToPrecision)
{
    using OtherVector = typename gko::distributed::Vector<mixed_type>;
    auto tmp = OtherVector::create(ref, comm);
    auto dtmp = OtherVector::create(exec, comm);
    init_vectors();

    x->convert_to(tmp.get());
    dx->convert_to(dtmp.get());

    GKO_ASSERT_MTX_NEAR(tmp->get_local(), dtmp->get_local(),
                        r<value_type>::value);
}


TEST_F(VectorLocalOps, MovesToPrecision)
{
    using OtherVector = typename gko::distributed::Vector<mixed_type>;
    auto tmp = OtherVector::create(ref, comm);
    auto dtmp = OtherVector::create(exec, comm);
    init_vectors();

    x->move_to(tmp.get());
    dx->move_to(dtmp.get());

    GKO_ASSERT_MTX_NEAR(tmp->get_local(), dtmp->get_local(),
                        r<value_type>::value);
}


TEST_F(VectorLocalOps, ComputeAbsoluteSameAsLocal)
{
    init_vectors();

    auto abs = x->compute_absolute();
    auto dabs = dx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs->get_local(), dabs->get_local(),
                        r<value_type>::value);
}


TEST_F(VectorLocalOps, ComputeAbsoluteInplaceSameAsLocal)
{
    init_vectors();

    x->compute_absolute_inplace();
    dx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, MakeComplexSameAsLocal)
{
    init_vectors();
    init_complex_vectors();

    complex = x->make_complex();
    dcomplex = dx->make_complex();

    GKO_ASSERT_MTX_NEAR(complex->get_local(), dcomplex->get_local(),
                        r<value_type>::value);
}


TEST_F(VectorLocalOps, MakeComplexInplaceSameAsLocal)
{
    init_vectors();
    init_complex_vectors();

    x->make_complex(complex.get());
    dx->make_complex(dcomplex.get());

    GKO_ASSERT_MTX_NEAR(complex->get_local(), dcomplex->get_local(),
                        r<value_type>::value);
}


TEST_F(VectorLocalOps, GetRealSameAsLocal)
{
    init_vectors();
    init_complex_vectors();

    x = complex->get_real();
    dx = dcomplex->get_real();

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, GetRealInplaceSameAsLocal)
{
    init_vectors();
    init_complex_vectors();

    complex->get_real(x.get());
    dcomplex->get_real(dx.get());

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, GetImagSameAsLocal)
{
    init_vectors();
    init_complex_vectors();

    x = complex->get_imag();
    dx = dcomplex->get_imag();

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, GetImagInplaceSameAsLocal)
{
    init_vectors();
    init_complex_vectors();

    complex->get_imag(x.get());
    dcomplex->get_imag(dx.get());

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, FillSameAsLocal)
{
    init_vectors();
    auto value = gko::test::detail::get_rand_value<value_type>(
        std::normal_distribution<gko::remove_complex<value_type>>(), engine);

    x->fill(value);
    dx->fill(value);

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, ScaleSameAsLocal)
{
    init_vectors();

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, InvScaleSameAsLocal)
{
    init_vectors();

    x->inv_scale(alpha.get());
    dx->inv_scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, AddScaleSameAsLocal)
{
    init_vectors();

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, SubScaleSameAsLocal)
{
    init_vectors();

    x->sub_scaled(alpha.get(), y.get());
    dx->sub_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(x->get_local(), dx->get_local(), r<value_type>::value);
}


TEST_F(VectorLocalOps, CreateRealViewSameAsLocal)
{
    using real_type = gko::remove_complex<value_type>;
    init_vectors();

    auto rv = x->create_real_view();
    auto drv = dx->create_real_view();

    EXPECT_EQ(rv->get_size()[0], drv->get_size()[0]);
    EXPECT_EQ(rv->get_size()[1], drv->get_size()[1]);
    EXPECT_EQ(rv->get_const_local()->get_stride(),
              drv->get_const_local()->get_stride());
    GKO_ASSERT_MTX_NEAR(rv->get_const_local(), drv->get_const_local(), 0.);
}


}  // namespace

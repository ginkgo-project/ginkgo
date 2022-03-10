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
#include <ginkgo/core/distributed/matrix.hpp>
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


class Matrix : public ::testing::Test {
public:
    using value_type = float;
    using mixed_type = double;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using dist_mtx_type = gko::distributed::Matrix<value_type, local_index_type,
                                                   global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;

    Matrix()
        : ref(gko::ReferenceExecutor::create()),
          exec(),
          comm(MPI_COMM_WORLD),
          size{53, 53},
          num_rhs(11),
          logger(gko::share(HostToDeviceLogger::create(exec))),
          engine(42)
    {
        init_executor(ref, exec, comm);
        exec->add_logger(logger);

        mat = dist_mtx_type::create(ref, comm);
        dmat = dist_mtx_type::create(exec, comm);
        x = dist_vec_type::create(ref, comm);
        dx = dist_vec_type::create(exec, comm);
        y = dist_vec_type::create(ref, comm);
        dy = dist_vec_type::create(exec, comm);
        alpha = dense_type::create(ref);
        dalpha = dense_type::create(exec);
        beta = dense_type::create(ref);
        dbeta = dense_type::create(exec);

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

    void SetUp() override
    {
        ASSERT_EQ(comm.size(), 3);

        generate_matrix_pair(mat, dmat);
        generate_vector_pair(x, dx);
        generate_vector_pair(y, dy);
        generate_scalar_pair(alpha, dalpha);
        generate_scalar_pair(beta, dbeta);
    }


    void generate_matrix_pair(std::unique_ptr<dist_mtx_type>& host,
                              std::unique_ptr<dist_mtx_type>& device)
    {
        auto md = gko::test::generate_random_matrix_data<value_type,
                                                         global_index_type>(
            size[0], size[1],
            std::uniform_int_distribution<gko::size_type>(0, size[1]),
            std::normal_distribution<value_type>(), engine);
        host->read_distributed(md, part.get());
        device = gko::clone(exec, host);
    }


    void generate_vector_pair(std::unique_ptr<dist_vec_type>& host,
                              std::unique_ptr<dist_vec_type>& device)
    {
        auto md = gko::test::generate_random_matrix_data<value_type,
                                                         global_index_type>(
            size[0], num_rhs,
            std::uniform_int_distribution<gko::size_type>(num_rhs, num_rhs),
            std::normal_distribution<value_type>(), engine);
        host->read_distributed(md, part.get());
        device = gko::clone(exec, host);
    }

    void generate_scalar_pair(std::unique_ptr<dense_type>& host,
                              std::unique_ptr<dense_type>& device)
    {
        host = gko::test::generate_random_matrix<dense_type>(
            1, 1, std::uniform_int_distribution<gko::size_type>(1, 1),
            std::normal_distribution<value_type>(), engine, ref);
        device = gko::clone(exec, host);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    gko::mpi::communicator comm;

    gko::dim<2> size;
    gko::size_type num_rhs;

    std::unique_ptr<part_type> part;

    std::unique_ptr<dist_mtx_type> mat;
    std::unique_ptr<dist_mtx_type> dmat;

    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> dx;
    std::unique_ptr<dist_vec_type> y;
    std::unique_ptr<dist_vec_type> dy;
    std::unique_ptr<dense_type> alpha;
    std::unique_ptr<dense_type> dalpha;
    std::unique_ptr<dense_type> beta;
    std::unique_ptr<dense_type> dbeta;

    std::shared_ptr<HostToDeviceLogger> logger;

    std::default_random_engine engine;
};


TEST_F(Matrix, ConvertsToPrecisionIsSameAsRef)
{
    using OtherMatrix =
        typename gko::distributed::Matrix<mixed_type, local_index_type,
                                          global_index_type>;
    auto tmp = OtherMatrix::create(ref, comm);
    auto dtmp = OtherMatrix::create(exec, comm);

    mat->convert_to(tmp.get());
    dmat->convert_to(dtmp.get());

    GKO_ASSERT_MTX_NEAR(tmp->get_local_diag(), dtmp->get_local_diag(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(tmp->get_local_offdiag(), dtmp->get_local_offdiag(),
                        r<value_type>::value);
}


TEST_F(Matrix, MovesToPrecisionIsSameAsRef)
{
    using OtherMatrix =
        typename gko::distributed::Matrix<mixed_type, local_index_type,
                                          global_index_type>;
    auto tmp = OtherMatrix::create(ref, comm);
    auto dtmp = OtherMatrix::create(exec, comm);

    mat->move_to(tmp.get());
    dmat->move_to(dtmp.get());

    GKO_ASSERT_MTX_NEAR(tmp->get_local_diag(), dtmp->get_local_diag(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(tmp->get_local_offdiag(), dtmp->get_local_offdiag(),
                        r<value_type>::value);
}


TEST_F(Matrix, ApplyIsSameAsRef)
{
    mat->apply(x.get(), y.get());
    dmat->apply(dx.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(y->get_local_vector(), dy->get_local_vector(), r<value_type>::value);
}


TEST_F(Matrix, AdvancedApplyIsSameAsRef)
{
    mat->apply(alpha.get(), x.get(), beta.get(), y.get());
    dmat->apply(dalpha.get(), dx.get(), dbeta.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(y->get_local_vector(), dy->get_local_vector(), r<value_type>::value);
}


TEST_F(Matrix, ApplyCopiesToHostOnlyIfNecessary)
{
    auto transfer_count_before = logger->get_transfer_count();

    dmat->apply(dx.get(), dy.get());

    ASSERT_EQ(logger->get_transfer_count() > transfer_count_before,
              needs_transfers(exec));
}


TEST_F(Matrix, AdvancedApplyCopiesToHostOnlyIfNecessary)
{
    auto transfer_count_before = logger->get_transfer_count();

    dmat->apply(dalpha.get(), dx.get(), dbeta.get(), dy.get());

    ASSERT_EQ(logger->get_transfer_count() > transfer_count_before,
              needs_transfers(exec));
}


}  // namespace

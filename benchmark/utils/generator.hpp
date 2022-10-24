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

#ifndef GINKGO_BENCHMARK_UTILS_GENERATOR_HPP
#define GINKGO_BENCHMARK_UTILS_GENERATOR_HPP


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#if GINKGO_BUILD_MPI
#include "benchmark/utils/distributed_helpers.hpp"
#endif


struct DefaultSystemGenerator {
    using Vec = vec<etype>;

    std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec,
        rapidjson::Value& config) const
    {
        std::ifstream mtx_fd(config["filename"].GetString());
        auto data = gko::read_generic_raw<etype, itype>(mtx_fd);
        return generate_matrix(std::move(exec), config, data);
    }

    std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec, rapidjson::Value& config,
        const gko::matrix_data<etype, itype>& data) const
    {
        return gko::share(::formats::matrix_factory(
            config["optimal"]["spmv"].GetString(), std::move(exec), data));
    }

    std::unique_ptr<Vec> generate_rhs(std::shared_ptr<const gko::Executor> exec,
                                      const gko::LinOp* system_matrix,
                                      rapidjson::Value& config) const
    {
        if (config.HasMember("rhs")) {
            std::ifstream rhs_fd{config["rhs"].GetString()};
            return gko::read<Vec>(rhs_fd, std::move(exec));
        } else {
            return ::generate_rhs(std::move(exec), system_matrix, engine);
        }
    }

    std::unique_ptr<Vec> generate_initial_guess(
        std::shared_ptr<const gko::Executor> exec,
        const gko::LinOp* system_matrix, const Vec* rhs) const
    {
        return ::generate_initial_guess(std::move(exec), system_matrix, rhs,
                                        engine);
    }

    std::unique_ptr<Vec> initialize(
        std::initializer_list<etype> il,
        std::shared_ptr<const gko::Executor> exec) const
    {
        return gko::initialize<Vec>(std::move(il), std::move(exec));
    }

    std::default_random_engine engine = get_engine();
};


#if GINKGO_BUILD_MPI


struct DistributedDefaultSystemGenerator {
    using Mtx = dist_mtx<etype, itype, gko::int64>;
    using Vec = dist_vec<etype>;

    std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec,
        rapidjson::Value& config) const
    {
        auto data = ::generate_matrix_data<etype, gko::int64>(config, comm);
        return generate_matrix(std::move(exec), config, data);
    }

    std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec, rapidjson::Value& config,
        const gko::matrix_data<etype, gko::int64>& data) const
    {
        auto part = gko::distributed::Partition<itype, gko::int64>::
            build_from_global_size_uniform(
                exec, comm.size(), static_cast<gko::int64>(data.size[0]));
        return ::create_distributed_matrix(
            exec, comm, config["optimal"]["spmv"]["local"].GetString(),
            config["optimal"]["spmv"]["non-local"].GetString(), data,
            part.get());
    }

    std::unique_ptr<Vec> generate_rhs(std::shared_ptr<const gko::Executor> exec,
                                      const gko::LinOp* system_matrix,
                                      rapidjson::Value& config) const
    {
        return Vec::create(
            exec, comm, gko::dim<2>{system_matrix->get_size()[0], FLAGS_nrhs},
            gko::as<DefaultSystemGenerator::Vec>(
                local_generator.generate_rhs(
                    exec, gko::as<Mtx>(system_matrix)->get_local_matrix().get(),
                    config))
                .get());
    }

    std::unique_ptr<Vec> generate_initial_guess(
        std::shared_ptr<const gko::Executor> exec,
        const gko::LinOp* system_matrix, const Vec* rhs) const
    {
        return Vec::create(
            exec, comm, gko::dim<2>{rhs->get_size()[0], FLAGS_nrhs},
            gko::as<DefaultSystemGenerator::Vec>(
                local_generator.generate_initial_guess(
                    exec, gko::as<Mtx>(system_matrix)->get_local_matrix().get(),
                    rhs->get_local_vector()))
                .get());
    }

    std::unique_ptr<Vec> initialize(
        std::initializer_list<etype> il,
        std::shared_ptr<const gko::Executor> exec) const
    {
        auto local = gko::initialize<DefaultSystemGenerator::Vec>(
            std::move(il), std::move(exec));
        auto global_rows = local->get_size()[0];
        comm.all_reduce(gko::ReferenceExecutor::create(), &global_rows, 1,
                        MPI_SUM);
        return Vec::create(exec, comm,
                           gko::dim<2>{global_rows, local->get_size()[1]},
                           local.get());
    }

    gko::mpi::communicator comm;
    DefaultSystemGenerator local_generator{};
};


#endif
#endif  // GINKGO_BENCHMARK_UTILS_GENERATOR_HPP

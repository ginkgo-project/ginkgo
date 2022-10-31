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


#include <string>


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#if GINKGO_BUILD_MPI
#include "benchmark/utils/distributed_helpers.hpp"
#endif


struct DefaultSystemGenerator {
    using IndexType = itype;
    using Vec = vec<etype>;

    gko::matrix_data<etype, IndexType> generate_matrix_data(
        rapidjson::Value& config) const
    {
        std::ifstream mtx_fd(config["filename"].GetString());
        return gko::read_generic_raw<etype, itype>(mtx_fd);
    }

    std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec,
        rapidjson::Value& config) const
    {
        auto data = generate_matrix_data(config);
        return generate_matrix_with_format(
            std::move(exec), config["optimal"]["spmv"].GetString(), data);
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<const gko::Executor> exec,
        const std::string& format_name,
        const gko::matrix_data<etype, itype>& data) const
    {
        return gko::share(
            ::formats::matrix_factory(format_name, std::move(exec), data));
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_default_format(
        std::shared_ptr<const gko::Executor> exec,
        const gko::matrix_data<etype, itype>& data) const
    {
        return generate_matrix_with_format(std::move(exec), "coo", data);
    }

    std::unique_ptr<Vec> create_matrix(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        etype value) const
    {
        auto res = Vec::create(exec);
        res->read(gko::matrix_data<etype, itype>(size, value));
        return res;
    }

    // creates a random matrix
    std::unique_ptr<Vec> create_matrix_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size) const
    {
        auto res = Vec::create(exec);
        res->read(gko::matrix_data<etype, itype>(
            size,
            std::uniform_real_distribution<gko::remove_complex<etype>>(-1.0,
                                                                       1.0),
            get_engine()));
        return res;
    }

    // creates a zero vector
    std::unique_ptr<Vec> create_vector(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size) const
    {
        auto res = Vec::create(exec, gko::dim<2>{size, 1});
        return res;
    }

    // creates a random vector
    std::unique_ptr<Vec> create_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size) const
    {
        return create_matrix_random(exec, gko::dim<2>{size, 1});
    }

    std::unique_ptr<Vec> initialize(
        std::initializer_list<etype> il,
        std::shared_ptr<const gko::Executor> exec) const
    {
        return gko::initialize<Vec>(std::move(il), std::move(exec));
    }
};


#if GINKGO_BUILD_MPI


template <typename LocalGeneratorType>
struct DistributedDefaultSystemGenerator {
    using LocalGenerator = LocalGeneratorType;

    using IndexType = gko::int64;

    using Mtx = dist_mtx<etype, itype, gko::int64>;
    using Vec = dist_vec<etype>;

    gko::matrix_data<etype, IndexType> generate_matrix_data(
        rapidjson::Value& config) const
    {
        return ::generate_matrix_data<etype, IndexType>(config, comm);
    }

    std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec,
        rapidjson::Value& config) const
    {
        auto data = generate_matrix_data(config);
        return generate_matrix_with_format(
            std::move(exec), config["optimal"]["spmv"].GetString(), data);
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<const gko::Executor> exec,
        const std::string& format_name,
        const gko::matrix_data<etype, IndexType>& data) const
    {
        auto part = gko::distributed::Partition<itype, gko::int64>::
            build_from_global_size_uniform(
                exec, comm.size(), static_cast<gko::int64>(data.size[0]));
        auto formats = split(format_name, '-');
        return ::create_distributed_matrix(exec, comm, formats[0], formats[1],
                                           data, part.get());
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_default_format(
        std::shared_ptr<const gko::Executor> exec,
        const gko::matrix_data<etype, gko::int64>& data) const
    {
        return generate_matrix_with_format(std::move(exec), "coo-coo", data);
    }

    std::unique_ptr<Vec> create_matrix(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        etype value) const
    {
        auto part = gko::distributed::Partition<itype, gko::int64>::
            build_from_global_size_uniform(exec, comm.size(),
                                           static_cast<gko::int64>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator
                .create_matrix(
                    exec,
                    gko::dim<2>{static_cast<gko::size_type>(
                                    part->get_part_size(comm.rank())),
                                size[1]},
                    value)
                .get());
    }

    // creates a random matrix
    std::unique_ptr<Vec> create_matrix_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size) const
    {
        auto part = gko::distributed::Partition<itype, gko::int64>::
            build_from_global_size_uniform(exec, comm.size(),
                                           static_cast<gko::int64>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator
                .create_matrix_random(
                    exec, gko::dim<2>{static_cast<gko::size_type>(
                                          part->get_part_size(comm.rank())),
                                      size[1]})
                .get());
    }

    // creates a zero vector
    std::unique_ptr<Vec> create_vector(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size) const
    {
        return create_matrix(exec, gko::dim<2>{size, 1}, etype{});
    }

    // creates a random vector
    std::unique_ptr<Vec> create_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size) const
    {
        return create_matrix_random(exec, gko::dim<2>{size, 1});
    }

    std::unique_ptr<Vec> initialize(
        std::initializer_list<etype> il,
        std::shared_ptr<const gko::Executor> exec) const
    {
        auto local = gko::initialize<typename LocalGenerator::Vec>(
            std::move(il), std::move(exec));
        auto global_rows = local->get_size()[0];
        comm.all_reduce(gko::ReferenceExecutor::create(), &global_rows, 1,
                        MPI_SUM);
        return Vec::create(exec, comm,
                           gko::dim<2>{global_rows, local->get_size()[1]},
                           local.get());
    }

    gko::mpi::communicator comm;
    LocalGenerator local_generator{};
};


#endif
#endif  // GINKGO_BENCHMARK_UTILS_GENERATOR_HPP

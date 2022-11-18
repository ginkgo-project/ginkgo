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


template <typename ValueType = etype, typename IndexType = itype>
struct DefaultSystemGenerator {
    using index_type = IndexType;
    using value_type = ValueType;
    using Vec = vec<ValueType>;

    static gko::matrix_data<ValueType, IndexType> generate_matrix_data(
        rapidjson::Value& config)
    {
        if (config.HasMember("filename")) {
            std::ifstream in(config["filename"].GetString());
            return gko::read_generic_raw<ValueType, IndexType>(in);
        } else if (config.HasMember("stencil")) {
            return generate_stencil<ValueType, IndexType>(
                config["stencil"].GetString(), config["size"].GetInt64());
        } else {
            throw std::runtime_error(
                "No known way to generate matrix data found.");
        }
    }

    static std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec, rapidjson::Value& config)
    {
        auto data = generate_matrix_data(config);
        return generate_matrix_with_format(
            std::move(exec), config["optimal"]["spmv"].GetString(), data);
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<const gko::Executor> exec,
        const std::string& format_name,
        const gko::matrix_data<ValueType, itype>& data)
    {
        return gko::share(
            ::formats::matrix_factory(format_name, std::move(exec), data));
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_default_format(
        std::shared_ptr<const gko::Executor> exec,
        const gko::matrix_data<ValueType, itype>& data)
    {
        return generate_matrix_with_format(std::move(exec), "coo", data);
    }

    static std::unique_ptr<Vec> create_multi_vector(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        ValueType value)
    {
        auto res = Vec::create(exec);
        res->read(gko::matrix_data<ValueType, itype>(size, value));
        return res;
    }

    static std::unique_ptr<Vec> create_multi_vector(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        gko::size_type stride)
    {
        auto res = Vec::create(exec, size, stride);
        return res;
    }

    // creates a random multi_vector
    static std::unique_ptr<Vec> create_multi_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
    {
        auto res = Vec::create(exec);
        res->read(gko::matrix_data<ValueType, itype>(
            size,
            std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1.0,
                                                                           1.0),
            get_engine()));
        return res;
    }

    // creates a zero vector
    static std::unique_ptr<Vec> create_vector(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size)
    {
        return create_multi_vector(std::move(exec), gko::dim<2>{size, 1}, 1);
    }

    // creates a random vector
    static std::unique_ptr<Vec> create_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size)
    {
        return create_multi_vector_random(exec, gko::dim<2>{size, 1});
    }

    static std::unique_ptr<Vec> initialize(
        std::initializer_list<ValueType> il,
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::initialize<Vec>(std::move(il), std::move(exec));
    }
};


#if GINKGO_BUILD_MPI


template <typename LocalGeneratorType>
struct DistributedDefaultSystemGenerator {
    using LocalGenerator = LocalGeneratorType;

    using index_type = gko::int64;
    using local_index_type = typename LocalGenerator::index_type;
    using value_type = typename LocalGenerator::value_type;

    using Mtx = dist_mtx<value_type, local_index_type, index_type>;
    using Vec = dist_vec<value_type>;

    gko::matrix_data<value_type, index_type> generate_matrix_data(
        rapidjson::Value& config) const
    {
        if (config.HasMember("filename")) {
            std::ifstream in(config["filename"].GetString());
            return gko::read_generic_raw<value_type, index_type>(in);
        } else if (config.HasMember("stencil")) {
            return generate_stencil<value_type, index_type>(
                config["stencil"].GetString(), comm, config["size"].GetInt64(),
                config["comm_pattern"].GetString() == std::string("optimal"));
        } else {
            throw std::runtime_error(
                "No known way to generate matrix data found.");
        }
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
        const gko::matrix_data<value_type, index_type>& data) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, gko::int64>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<gko::int64>(data.size[0]));
        auto formats = split(format_name, '-');
        return ::create_distributed_matrix(exec, comm, formats[0], formats[1],
                                           data, part.get());
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_default_format(
        std::shared_ptr<const gko::Executor> exec,
        const gko::matrix_data<value_type, gko::int64>& data) const
    {
        return generate_matrix_with_format(std::move(exec), "coo-coo", data);
    }

    std::unique_ptr<Vec> create_multi_vector(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        value_type value) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, gko::int64>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<gko::int64>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator
                .create_multi_vector(
                    exec,
                    gko::dim<2>{static_cast<gko::size_type>(
                                    part->get_part_size(comm.rank())),
                                size[1]},
                    value)
                .get());
    }

    // creates a random multi_vector
    std::unique_ptr<Vec> create_multi_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, gko::int64>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<gko::int64>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator
                .create_multi_vector_random(
                    exec, gko::dim<2>{static_cast<gko::size_type>(
                                          part->get_part_size(comm.rank())),
                                      size[1]})
                .get());
    }

    // creates a zero vector
    std::unique_ptr<Vec> create_vector(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size) const
    {
        return create_multi_vector(exec, gko::dim<2>{size, 1}, value_type{});
    }

    // creates a random vector
    std::unique_ptr<Vec> create_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size) const
    {
        return create_multi_vector_random(exec, gko::dim<2>{size, 1});
    }

    std::unique_ptr<Vec> initialize(
        std::initializer_list<value_type> il,
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

    gko::experimental::mpi::communicator comm;
    LocalGenerator local_generator{};
};


#endif
#endif  // GINKGO_BENCHMARK_UTILS_GENERATOR_HPP

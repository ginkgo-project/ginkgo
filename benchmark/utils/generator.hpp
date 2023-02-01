/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include "benchmark/utils/stencil_matrix.hpp"
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

    static std::shared_ptr<gko::LinOp> generate_matrix_with_optimal_format(
        std::shared_ptr<gko::Executor> exec, rapidjson::Value& config)
    {
        auto data = generate_matrix_data(config);
        return generate_matrix_with_format(
            std::move(exec), config["optimal"]["spmv"].GetString(), data);
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<gko::Executor> exec, const std::string& format_name,
        const gko::matrix_data<ValueType, itype>& data,
        rapidjson::Value* spmv_case = nullptr,
        rapidjson::MemoryPoolAllocator<>* allocator = nullptr)
    {
        auto storage_logger = std::make_shared<StorageLogger>();
        if (spmv_case && allocator) {
            exec->add_logger(storage_logger);
        }

        auto mtx =
            gko::share(::formats::matrix_factory(format_name, exec, data));

        if (spmv_case && allocator) {
            exec->remove_logger(storage_logger);
            storage_logger->write_data(*spmv_case, *allocator);
        }

        return mtx;
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_default_format(
        std::shared_ptr<gko::Executor> exec,
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

    static std::unique_ptr<Vec> create_multi_vector_strided(
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

    using index_type = global_itype;
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
            auto local_size = static_cast<global_itype>(
                config["size"].GetInt64() / comm.size());
            return generate_stencil<value_type, index_type>(
                config["stencil"].GetString(), comm, local_size,
                config["comm_pattern"].GetString() == std::string("optimal"));
        } else {
            throw std::runtime_error(
                "No known way to generate matrix data found.");
        }
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_optimal_format(
        std::shared_ptr<gko::Executor> exec, rapidjson::Value& config) const
    {
        auto data = generate_matrix_data(config);
        return generate_matrix_with_format(
            std::move(exec), config["optimal"]["spmv"].GetString(), data);
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<gko::Executor> exec, const std::string& format_name,
        const gko::matrix_data<value_type, index_type>& data,
        rapidjson::Value* spmv_case = nullptr,
        rapidjson::MemoryPoolAllocator<>* allocator = nullptr) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, global_itype>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<global_itype>(data.size[0]));
        auto formats = split(format_name, '-');

        auto local_mat = formats::matrix_type_factory.at(formats[0])(exec);
        auto non_local_mat = formats::matrix_type_factory.at(formats[1])(exec);

        auto storage_logger = std::make_shared<StorageLogger>();
        if (spmv_case && allocator) {
            exec->add_logger(storage_logger);
        }

        auto dist_mat = dist_mtx<etype, itype, global_itype>::create(
            exec, comm, local_mat.get(), non_local_mat.get());
        dist_mat->read_distributed(data, part.get());

        if (spmv_case && allocator) {
            exec->remove_logger(storage_logger);
            storage_logger->write_data(comm, *spmv_case, *allocator);
        }

        return dist_mat;
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_default_format(
        std::shared_ptr<gko::Executor> exec,
        const gko::matrix_data<value_type, global_itype>& data) const
    {
        return generate_matrix_with_format(std::move(exec), "coo-coo", data);
    }

    std::unique_ptr<Vec> create_multi_vector(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        value_type value) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, global_itype>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<global_itype>(size[0]));
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

    std::unique_ptr<Vec> create_multi_vector_strided(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        gko::size_type stride) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, global_itype>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<global_itype>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator
                .create_multi_vector_strided(
                    exec,
                    gko::dim<2>{static_cast<gko::size_type>(
                                    part->get_part_size(comm.rank())),
                                size[1]},
                    stride)
                .get());
    }

    // creates a random multi_vector
    std::unique_ptr<Vec> create_multi_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, global_itype>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<global_itype>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator
                .create_multi_vector_random(
                    exec, gko::dim<2>{static_cast<gko::size_type>(
                                          part->get_part_size(comm.rank())),
                                      size[1]})
                .get());
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
